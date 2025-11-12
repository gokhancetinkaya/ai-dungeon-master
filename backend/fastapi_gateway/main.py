"""
FastAPI Gateway for AI Dungeon Master
Orchestrates text and image model inference via Triton
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from typing import AsyncGenerator
import logging
import json
from datetime import datetime
import uuid

from orchestrator import MultiModelOrchestrator
from models import ChatRequest, ChatResponse, HealthResponse, ModelStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Dungeon Master API",
    description="Multi-model orchestration for D&D text-based RPG",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = MultiModelOrchestrator()

# Store active sessions (in production, use Redis)
sessions = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Dungeon Master API...")
    await orchestrator.initialize()
    logger.info("Services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Dungeon Master API...")
    await orchestrator.cleanup()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "AI Dungeon Master API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        text_status = await orchestrator.check_text_health()
        image_status = await orchestrator.check_image_health()
        
        return HealthResponse(
            status="healthy" if text_status and image_status else "degraded",
            text_model_available=text_status,
            image_model_available=image_status,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            text_model_available=False,
            image_model_available=False,
            timestamp=datetime.utcnow().isoformat()
        )


@app.get("/api/models/status", response_model=ModelStatus, tags=["Models"])
async def get_model_status():
    """Get status of all models"""
    try:
        text_info = await orchestrator.get_text_info()
        image_info = await orchestrator.get_image_info()
        
        return ModelStatus(
            text=text_info,
            image_model=image_info
        )
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user message and returns narrative + image
    Supports streaming response
    """
    try:
        logger.info(f"Chat request from session {request.session_id}: {request.message}")
        
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = {
                "history": [],
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
        
        session = sessions[session_id]
        session["last_activity"] = datetime.utcnow()
        
        # Add user message to history
        session["history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Generate response using orchestrator
        result = await orchestrator.process_message(
            message=request.message,
            history=session["history"],
            stream=request.stream
        )
        
        # Add assistant response to history
        session["history"].append({
            "role": "assistant",
            "content": result["narrative"],
            "scene_description": result.get("scene_description"),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Return response
        return ChatResponse(
            session_id=session_id,
            narrative=result["narrative"],
            scene_description=result.get("scene_description"),
            image_url=result.get("image_url"),
            image_base64=result.get("image_base64"),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - streams narrative tokens as they're generated
    """
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            session_id = request.session_id or str(uuid.uuid4())
            
            async for chunk in orchestrator.stream_message(
                message=request.message,
                history=sessions.get(session_id, {}).get("history", [])
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "history": [],
        "created_at": datetime.utcnow()
    }
    
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            logger.info(f"WS message from {session_id}: {user_message}")
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": "Processing..."
            })
            
            # Stream narrative tokens
            full_narrative = ""
            async for chunk in orchestrator.stream_message(
                message=user_message,
                history=sessions[session_id]["history"]
            ):
                if chunk["type"] == "narrative_token":
                    await websocket.send_json(chunk)
                    full_narrative += chunk["content"]
                elif chunk["type"] == "scene_description":
                    await websocket.send_json(chunk)
                elif chunk["type"] == "image":
                    await websocket.send_json(chunk)
                elif chunk["type"] == "done":
                    await websocket.send_json(chunk)
            
            # Update session history
            sessions[session_id]["history"].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_narrative}
            ])
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        del sessions[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close()


@app.delete("/api/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear session history"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Get session history"""
    if session_id in sessions:
        return sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

