"""
Triton Inference Server clients for text and image models
"""

import logging
import numpy as np
from typing import AsyncGenerator, Dict
import base64
import io

logger = logging.getLogger(__name__)

# Import Triton client library
# This will fail if tritonclient is not installed (requires Triton Inference Server)
try:
    import tritonclient.grpc.aio as grpcclient  # type: ignore
    TRITON_AVAILABLE = True
except ImportError:
    logger.warning("Triton client library not available - Triton functionality disabled")
    TRITON_AVAILABLE = False


class TritonTextClient:
    """Client for text model inference via Triton"""
    
    def __init__(self, url: str, model_name: str = "mistral_dm"):
        """
        Initialize Triton text model client
        
        Args:
            url: Triton server URL (e.g., "localhost:8001" for gRPC)
            model_name: Name of the text model in Triton repository
        """
        self.url = url
        self.model_name = model_name
        self.client = None
        self.connected = False
    
    async def connect(self):
        """Connect to Triton server"""
        if not TRITON_AVAILABLE:
            logger.warning("Triton client not available, skipping connection")
            self.connected = False
            return
        
        try:
            self.client = grpcclient.InferenceServerClient(url=self.url)
            # Check if server is ready
            await self.client.is_server_ready()
            self.connected = True
            logger.info(f"Connected to Triton text model server at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton text model server: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from Triton server"""
        if self.client:
            await self.client.close()
            self.connected = False
    
    async def is_ready(self) -> bool:
        """Check if model is ready"""
        if not self.connected or not TRITON_AVAILABLE:
            return False
        
        try:
            return await self.client.is_model_ready(self.model_name)
        except Exception as e:
            logger.error(f"Error checking model readiness: {e}")
            return False
    
    async def get_model_info(self) -> Dict:
        """Get model metadata"""
        if not self.connected or not TRITON_AVAILABLE:
            return {"name": self.model_name, "backend": "mock"}
        
        try:
            metadata = await self.client.get_model_metadata(self.model_name)
            return {
                "name": metadata.name,
                "backend": metadata.platform,
                "version": metadata.versions[0] if metadata.versions else "1"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"name": self.model_name, "backend": "unknown"}
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        if not self.connected or not TRITON_AVAILABLE:
            raise RuntimeError("Text model not connected. Triton server may not be running or model not loaded.")
        
        try:
            # CLIENT → TRITON (Python → Numpy conversion)
            # Convert Python types to numpy arrays for gRPC transmission
            # Triton only understands numpy arrays, not Python native types
            
            # Prepare inputs for Triton
            # Model has batching disabled, so shapes are [data_size] not [batch, data_size]
            inputs = []
            # For STRING inputs without batching, shape is [num_strings]
            # Python string → bytes → numpy array
            inputs.append(grpcclient.InferInput("INPUT_TEXT", [1], "BYTES"))
            inputs[0].set_data_from_numpy(np.array([prompt.encode('utf-8')], dtype=np.object_))
            
            # Python int → numpy array
            inputs.append(grpcclient.InferInput("MAX_TOKENS", [1], "INT32"))
            inputs[1].set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
            
            # Python float → numpy array
            inputs.append(grpcclient.InferInput("TEMPERATURE", [1], "FP32"))
            inputs[2].set_data_from_numpy(np.array([temperature], dtype=np.float32))
            
            # Request outputs
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT_TEXT"))
            
            # Perform inference
            response = await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # TRITON → CLIENT (Numpy → Python conversion)
            # Convert numpy array back to Python string
            # Extract result (output shape is (1,) with the bytes string)
            # Numpy array → bytes → Python string
            output_text = response.as_numpy("OUTPUT_TEXT")
            return output_text[0].decode('utf-8')
            
        except Exception as e:
            logger.error(f"Text model generation error: {e}")
            raise RuntimeError(f"Text model generation failed: {e}") from e
    
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated tokens
        
        Note: Requires Triton model with streaming support
        """
        if not self.connected or not TRITON_AVAILABLE:
            raise RuntimeError("Text model not connected. Triton server may not be running or model not loaded.")
        
        # TODO: Implement actual streaming with Triton
        # For now, fallback to batch generation
        response = await self.generate(prompt, max_tokens, temperature)
        words = response.split()
        for word in words:
            yield word + " "
    


class TritonImageClient:
    """Client for image model inference via Triton"""
    
    def __init__(self, url: str, model_name: str = "sdxl_pov"):
        """
        Initialize Triton image model client
        
        Args:
            url: Triton server URL
            model_name: Name of the image model in Triton repository
        """
        self.url = url
        self.model_name = model_name
        self.client = None
        self.connected = False
    
    async def connect(self):
        """Connect to Triton server"""
        if not TRITON_AVAILABLE:
            logger.warning("Triton client not available, skipping connection")
            self.connected = False
            return
        
        try:
            self.client = grpcclient.InferenceServerClient(url=self.url)
            await self.client.is_server_ready()
            self.connected = True
            logger.info(f"Connected to Triton image model server at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton image model server: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from Triton server"""
        if self.client:
            await self.client.close()
            self.connected = False
    
    async def is_ready(self) -> bool:
        """Check if model is ready"""
        if not self.connected or not TRITON_AVAILABLE:
            return False
        
        try:
            return await self.client.is_model_ready(self.model_name)
        except Exception as e:
            logger.error(f"Error checking model readiness: {e}")
            return False
    
    async def get_model_info(self) -> Dict:
        """Get model metadata"""
        if not self.connected or not TRITON_AVAILABLE:
            return {"name": self.model_name, "backend": "mock"}
        
        try:
            metadata = await self.client.get_model_metadata(self.model_name)
            return {
                "name": metadata.name,
                "backend": metadata.platform,
                "version": metadata.versions[0] if metadata.versions else "1"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"name": self.model_name, "backend": "unknown"}
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 8,
        guidance_scale: float = 7.5,
    ) -> str:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
        
        Returns:
            Base64 encoded image
        """
        if not self.connected or not TRITON_AVAILABLE:
            raise RuntimeError("Image model not connected. Triton server may not be running or model not loaded.")
        
        try:
            # CLIENT → TRITON (Python → Numpy conversion)
            # Convert Python types to numpy arrays for gRPC transmission
            # Triton only understands numpy arrays, not Python native types
            
            # Prepare inputs for Triton
            # Model has batching disabled, so shapes are [data_size] not [batch, data_size]
            inputs = []
            # Python string → bytes → numpy array
            inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES"))
            inputs[0].set_data_from_numpy(np.array([prompt.encode('utf-8')], dtype=np.object_))
            
            # Python string → bytes → numpy array
            inputs.append(grpcclient.InferInput("NEGATIVE_PROMPT", [1], "BYTES"))
            inputs[1].set_data_from_numpy(np.array([negative_prompt.encode('utf-8')], dtype=np.object_))
            
            # Python int → numpy array
            inputs.append(grpcclient.InferInput("NUM_STEPS", [1], "INT32"))
            inputs[2].set_data_from_numpy(np.array([num_inference_steps], dtype=np.int32))
            
            # Python float → numpy array
            inputs.append(grpcclient.InferInput("GUIDANCE_SCALE", [1], "FP32"))
            inputs[3].set_data_from_numpy(np.array([guidance_scale], dtype=np.float32))
            
            # Request outputs
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("IMAGE"))
            
            # Perform inference
            response = await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # TRITON → CLIENT (Numpy → Python conversion)
            # Convert numpy array to base64-encoded image
            
            # Extract and encode image
            # Output has batch dimension [batch_size, C, H, W]
            # Step 1: Get numpy array from Triton (float32, [1, 3, 768, 768])
            raw_array = response.as_numpy("IMAGE")
            logger.info(f"Image array shape WITH batch: {raw_array.shape}, dtype: {raw_array.dtype}")
            image_array = raw_array[0]  # Remove batch dimension → [3, 768, 768]
            logger.info(f"Image array shape AFTER removing batch: {image_array.shape}, dtype: {image_array.dtype}")
            
            # Step 2: Convert from [C, H, W] to [H, W, C] for PIL
            image_array = np.transpose(image_array, (1, 2, 0))  # → [768, 768, 3]
            logger.info(f"Image array shape after transpose: {image_array.shape}")
            
            # Step 3: Convert from [0, 1] to [0, 255] and then to uint8
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
            
            # Step 4: Convert numpy array → PIL Image → base64 string
            from PIL import Image
            image = Image.fromarray(image_array)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            # Final: base64 string ready for JSON response
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise RuntimeError(f"Image generation failed: {e}") from e

