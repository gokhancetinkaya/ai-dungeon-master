"""
Multi-Model Orchestrator
Coordinates text and image model inference via Triton
"""

import logging
import re
import os
from typing import List, Dict, AsyncGenerator, Optional

from triton_client import TritonTextClient, TritonImageClient

logger = logging.getLogger(__name__)


class MultiModelOrchestrator:
    """
    Orchestrates inference across text and image models
    """
    
    def __init__(self):
        """Initialize orchestrator with Triton clients"""
        # Get Triton endpoints from environment or use defaults
        text_url = os.getenv("TRITON_TEXT_URL", "triton-text-service:8001")
        image_url = os.getenv("TRITON_IMAGE_URL", "triton-image-service:8001")
        
        self.text_client = TritonTextClient(url=text_url)
        self.image_client = TritonImageClient(url=image_url)
        
        # System prompt for the Dungeon Master
        self.system_prompt = """You are an experienced Dungeon Master for a D&D adventure. 
Your role is to create immersive, engaging narratives in response to player actions.

Guidelines:
- ONLY respond to the player's current action - do NOT generate future player actions or dialogue
- Be descriptive and atmospheric
- Respond to player actions with consequences
- Keep responses concise (2-4 sentences for the narrative)
- End your response with a scene description in square brackets: [SCENE: visual description]
- The SCENE description should be detailed and suitable for image generation (describe setting, lighting, mood, composition)
- Use a consistent cinematic fantasy RPG art style in scene descriptions
- Include details like lighting (torch light, moonlight, magical glow), atmosphere (misty, foggy, clear), and perspective (first-person view)
- STOP after the [SCENE] tag - do not continue the story

Example:
Player: "I open the door"
DM: "The heavy oak door creaks open, revealing a dimly lit corridor. Stone walls glisten with moisture, and the air smells of ancient dust. In the distance, you hear the echo of dripping water. [SCENE: first-person perspective of dark medieval stone corridor, warm torch light casting dancing shadows on wet stone walls, atmospheric fog, cinematic composition, high detail]"
"""
        
        # Consistent style prefix for all image generation
        self.image_style_prefix = "fantasy RPG game art, cinematic lighting, detailed digital painting, first-person perspective,"
        self.image_negative_prompt = "blurry, low quality, distorted, ugly, deformed, text, watermark, signature, cartoon, anime, 3d render, photorealistic, modern, contemporary"
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize connections to Triton servers"""
        try:
            logger.info("Initializing Triton clients...")
            await self.text_client.connect()
            await self.image_client.connect()
            self.initialized = True
            logger.info("Triton clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Triton clients: {e}")
            # In development, continue even if Triton is not available
            self.initialized = False
    
    async def cleanup(self):
        """Cleanup connections"""
        try:
            await self.text_client.disconnect()
            await self.image_client.disconnect()
            logger.info("Triton clients disconnected")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def check_text_health(self) -> bool:
        """Check text model service health"""
        try:
            return await self.text_client.is_ready()
        except Exception as e:
            logger.error(f"Text model health check failed: {e}")
            return False
    
    async def check_image_health(self) -> bool:
        """Check image model service health"""
        try:
            return await self.image_client.is_ready()
        except Exception as e:
            logger.error(f"Image model health check failed: {e}")
            return False
    
    async def get_text_info(self) -> Dict:
        """Get text model information"""
        try:
            info = await self.text_client.get_model_info()
            return {
                "name": info.get("name", "mistral_dm"),
                "status": "ready" if await self.check_text_health() else "unavailable",
                "backend": info.get("backend", "pytorch"),
                "version": info.get("version", "1")
            }
        except Exception as e:
            logger.error(f"Failed to get text model info: {e}")
            return {
                "name": "mistral_dm",
                "status": "unavailable",
                "backend": "unknown",
                "version": None
            }
    
    async def get_image_info(self) -> Dict:
        """Get image model information"""
        try:
            info = await self.image_client.get_model_info()
            return {
                "name": info.get("name", "sdxl_pov"),
                "status": "ready" if await self.check_image_health() else "unavailable",
                "backend": info.get("backend", "pytorch"),
                "version": info.get("version", "1")
            }
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {
                "name": "sdxl_pov",
                "status": "unavailable",
                "backend": "unknown",
                "version": None
            }
    
    def extract_scene_description(self, narrative: str) -> tuple[str, Optional[str]]:
        """
        Extract scene description from narrative
        Format: [SCENE: description]
        Returns: (narrative_without_scene, scene_description)
        """
        pattern = r'\[SCENE:\s*([^\]]+)\]'
        match = re.search(pattern, narrative, re.IGNORECASE)
        
        if match:
            scene_description = match.group(1).strip()
            # Cut off everything from [SCENE onwards to prevent hallucinated player actions
            scene_start = match.start()
            narrative_clean = narrative[:scene_start].strip()
            return narrative_clean, scene_description
        
        return narrative, None
    
    def build_prompt(self, message: str, history: List[Dict]) -> str:
        """Build prompt with system message and conversation history"""
        prompt = f"{self.system_prompt}\n\n"
        
        # Add conversation history (last 5 turns to keep context manageable)
        recent_history = history[-10:] if len(history) > 10 else history
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                prompt += f"Player: {content}\n"
            elif role == "assistant":
                prompt += f"DM: {content}\n"
        
        # Add current message
        prompt += f"Player: {message}\nDM: "
        
        return prompt
    
    async def process_message(
        self,
        message: str,
        history: List[Dict],
        stream: bool = False
    ) -> Dict:
        """
        Process user message:
        1. Generate narrative with text model
        2. Extract scene description
        3. Generate image from scene
        4. Return combined result
        """
        try:
            # Build prompt
            prompt = self.build_prompt(message, history)
            
            # Generate narrative with text model
            logger.info("Generating narrative with text model...")
            if self.initialized and await self.check_text_health():
                narrative = await self.text_client.generate(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.8
                )
            else:
                raise RuntimeError("Text model client not available")
            
            # Extract scene description
            narrative_clean, scene_description = self.extract_scene_description(narrative)
            
            logger.info(f"Narrative: {narrative_clean}")
            logger.info(f"Scene: {scene_description}")
            
            # Generate image if scene description exists
            image_base64 = None
            if scene_description:
                try:
                    logger.info("Generating scene image...")
                    if self.initialized and await self.check_image_health():
                        # Add consistent style prefix to ensure coherent art style
                        styled_prompt = f"{self.image_style_prefix} {scene_description}"
                        logger.info(f"Styled prompt: {styled_prompt}")
                        
                        image_base64 = await self.image_client.generate(
                            prompt=styled_prompt,
                            negative_prompt=self.image_negative_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5
                        )
                    else:
                        logger.warning("Image model client not available, skipping image generation")
                        image_base64 = None
                except Exception as e:
                    logger.error(f"Image generation failed: {e}")
                    image_base64 = None
            
            return {
                "narrative": narrative_clean,
                "scene_description": scene_description,
                "image_base64": image_base64
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            raise
    
    async def stream_message(
        self,
        message: str,
        history: List[Dict]
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream response:
        1. Stream narrative tokens as they're generated
        2. Send scene description when extracted
        3. Generate and send image
        """
        try:
            # Build prompt
            prompt = self.build_prompt(message, history)
            
            # Stream narrative tokens
            full_narrative = ""
            async for token in self.text_client.stream_generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.8
            ):
                full_narrative += token
                yield {
                    "type": "narrative_token",
                    "content": token
                }
            
            # Extract scene description
            narrative_clean, scene_description = self.extract_scene_description(full_narrative)
            
            # Yield the clean narrative (without [SCENE] tag)
            yield {
                "type": "narrative_complete",
                "content": narrative_clean
            }
            
            if scene_description:
                yield {
                    "type": "scene_description",
                    "content": scene_description
                }
                
                # Generate image with consistent style
                try:
                    if self.initialized and await self.check_image_health():
                        styled_prompt = f"{self.image_style_prefix} {scene_description}"
                        image_base64 = await self.image_client.generate(
                            prompt=styled_prompt,
                            negative_prompt=self.image_negative_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.5
                        )
                        
                        yield {
                            "type": "image",
                            "image_base64": image_base64
                        }
                except Exception as e:
                    logger.error(f"Image generation failed: {e}")
            
            yield {"type": "done"}
            
        except Exception as e:
            logger.error(f"Error streaming message: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}

