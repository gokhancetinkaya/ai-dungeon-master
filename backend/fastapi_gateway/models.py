"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str = Field(..., description="User's message/action")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    stream: bool = Field(False, description="Enable streaming response")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I open the ancient wooden door",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Chat response with narrative and image"""
    session_id: str = Field(..., description="Session ID")
    narrative: str = Field(..., description="DM's narrative response")
    scene_description: Optional[str] = Field(None, description="Extracted scene description for image")
    image_url: Optional[str] = Field(None, description="URL to generated image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "narrative": "The heavy oak door creaks open...",
                "scene_description": "dark medieval corridor with flickering torches",
                "image_url": "http://example.com/image.png",
                "timestamp": "2025-10-24T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    text_model_available: bool = Field(..., description="Text model availability")
    image_model_available: bool = Field(..., description="Image model availability")
    timestamp: str = Field(..., description="Check timestamp")


class ModelInfo(BaseModel):
    """Information about a model"""
    name: str
    status: str
    backend: str
    version: Optional[str] = None


class ModelStatus(BaseModel):
    """Status of all models"""
    text_model: ModelInfo
    image_model: ModelInfo


class StreamChunk(BaseModel):
    """Chunk of streamed response"""
    type: str = Field(..., description="Type: narrative_token, scene_description, image, done")
    content: Optional[str] = Field(None, description="Content of the chunk")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image if type is image")

