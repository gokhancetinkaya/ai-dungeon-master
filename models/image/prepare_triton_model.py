#!/usr/bin/env python3
"""
Prepare image generation model for Triton deployment
"""

import argparse
import shutil
from pathlib import Path

MODEL_PY_TEMPLATE = '''"""
Stable Diffusion XL image generation model for Triton Inference Server
"""

import io
import base64
import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Python backend wrapper for image generation model"""
    
    def initialize(self, args):
        """Load image generation model"""
        # Triton passes the model instance directory (e.g., /models/sdxl_pov/1/)
        # Model files are in the same directory as model.py
        import os
        self.model_dir = os.path.dirname(os.path.realpath(__file__))
        
        print(f"Loading image generation model from {{self.model_dir}}")
        
        # Load pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16
        )
        
        # Enable CPU offloading only if in shared GPU mode
        # Single GPU: ENABLE_CPU_OFFLOAD=true (default) - saves VRAM
        # Dual GPU: ENABLE_CPU_OFFLOAD=false - better performance
        enable_offloading = os.environ.get("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
        if enable_offloading:
            print("Enabling CPU offloading (shared GPU mode - reduces VRAM to 1-4GB peak)")
            # This moves model components between GPU and CPU as needed
            self.pipe.enable_model_cpu_offload()
        else:
            print("Keeping full model on GPU (dedicated GPU mode - 7GB VRAM, faster inference)")
            self.pipe.to("cuda")
        
        # Enable optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not available, skipping")
        
        print("Image generation model loaded successfully")
    
    def execute(self, requests):
        """Execute image generation inference"""
        responses = []
        
        for request in requests:
            # INPUT: Triton → Model (Numpy → Tensor conversion)
            # 1. Get input from Triton (numpy array)
            prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            # 2. Convert numpy → Python string
            prompt = prompt.as_numpy()[0].decode('utf-8')
            
            # Get optional parameters
            try:
                negative_prompt = pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                negative_prompt = negative_prompt.as_numpy()[0].decode('utf-8')
            except:
                negative_prompt = "blurry, low quality, distorted"
            
            try:
                num_steps = pb_utils.get_input_tensor_by_name(request, "NUM_STEPS")
                num_steps = int(num_steps.as_numpy()[0])
            except:
                num_steps = 8
            
            try:
                guidance = pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                guidance = float(guidance.as_numpy()[0])
            except:
                guidance = 7.5
            
            # Generate image (Pipeline internally uses PyTorch tensors)
            with torch.no_grad():
                # 3. Pipeline converts string → PyTorch tensors internally
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    height=768,
                    width=768
                ).images[0]  # Returns PIL Image
            
            # OUTPUT: Model → Triton (Tensor → Numpy conversion)
            # 4. Convert PIL Image → numpy array
            # Ensure image is RGB (some models may return grayscale)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension explicitly for Triton
            # Shape: (C, H, W) -> (1, C, H, W)
            image_array = np.expand_dims(image_array, axis=0)
            
            # Free GPU memory after generation
            torch.cuda.empty_cache()
            
            # 5. Send numpy array to Triton
            output_tensor = pb_utils.Tensor(
                "IMAGE",
                image_array
            )
            
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        
        return responses
    
    def finalize(self):
        """Cleanup"""
        print("Cleaning up image generation model")
        del self.pipe
'''

CONFIG_PBTXT_TEMPLATE = '''name: "{model_name}"
backend: "python"

input [
  {{
    name: "PROMPT"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }},
  {{
    name: "NEGATIVE_PROMPT"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "NUM_STEPS"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "GUIDANCE_SCALE"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "IMAGE"
    data_type: TYPE_FP32
    dims: [ 3, 768, 768 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''


def prepare_model(model_path: str, output_dir: str, model_name: str = "sdxl_pov"):
    """Prepare image generation model repository for Triton"""
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        print(f"Error: Model path not found: {{model_path}}")
        exit(1)
    
    print(f"Preparing Triton model repository for image generation model...")
    print(f"Source: {{model_path}}")
    print(f"Destination: {{output_dir}}/{{model_name}}")
    
    # Create model repository structure
    model_dir = output_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Move model files (to avoid using 2x disk space)
    print("\n[1/3] Moving model files...")
    for item in model_path.iterdir():
        dest = version_dir / item.name
        shutil.move(str(item), str(dest))
    print("✓ Model files moved")
    
    # Create model.py
    print("\n[2/3] Creating Python backend wrapper...")
    (version_dir / "model.py").write_text(MODEL_PY_TEMPLATE)
    print("✓ model.py created")
    
    # Create config.pbtxt
    print("\n[3/3] Creating Triton configuration...")
    config_content = CONFIG_PBTXT_TEMPLATE.format(model_name=model_name)
    (model_dir / "config.pbtxt").write_text(config_content)
    print("✓ config.pbtxt created")
    
    # Summary
    print("\n" + "="*50)
    print("Model Repository Ready!")
    print("="*50)
    print(f"Location: {{model_dir}}")
    print(f"\nStructure:")
    print(f"{{model_name}}/")
    print(f"├── config.pbtxt")
    print(f"└── 1/")
    print(f"    ├── model.py")
    print(f"    └── [image generation model components]")

def main():
    parser = argparse.ArgumentParser(description="Prepare image generation model for Triton")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to downloaded model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./triton_model_repo",
        help="Output directory (default: ./triton_model_repo)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sdxl_pov",
        help="Model name in Triton (default: sdxl_pov)"
    )
    
    args = parser.parse_args()
    
    prepare_model(args.model_path, args.output, args.model_name)


if __name__ == "__main__":
    main()

