#!/usr/bin/env python3
"""
Prepare text generation model for Triton deployment
"""

import argparse
import shutil
from pathlib import Path

DUNGEON_MASTER_PROMPT = """You are an experienced Dungeon Master for a D&D adventure.
Your role is to create immersive, engaging narratives in response to player actions.

Guidelines:
- Be descriptive and atmospheric
- Respond to player actions with consequences
- Keep responses concise (2-4 sentences)
- At the end of each response, add a scene description in square brackets like: [SCENE: description]
- The SCENE description should be visual and suitable for image generation (50 words max)

Example:
Player: "I open the door"
You: "The heavy oak door creaks open, revealing a dimly lit corridor. Stone walls glisten with moisture, and the air smells of ancient dust. In the distance, you hear the echo of dripping water. [SCENE: first-person view of dark medieval stone corridor, flickering torch light on damp walls, mysterious shadows]"
"""

MODEL_PY_TEMPLATE = '''"""
Mistral-7B text generation model for Triton Inference Server
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Python backend wrapper for text generation model"""
    
    def initialize(self, args):
        """Load text generation model and tokenizer"""
        # Triton passes the model instance directory (e.g., /models/mistral_dm/1/)
        # Model files are in the same directory as model.py
        import os
        self.model_dir = os.path.dirname(os.path.realpath(__file__))
        
        print(f"Loading text generation model from {{self.model_dir}}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically place model layers on GPU/CPU based on available memory
        )
        self.model.eval()
        
        # System prompt
        self.system_prompt = """{system_prompt}"""
        
        print("Text generation model loaded successfully")
    
    def execute(self, requests):
        """Execute text generation inference"""
        responses = []
        
        for request in requests:
            # INPUT: Triton → Model (Numpy → Tensor conversion)
            # 1. Get input from Triton (numpy array)
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # 2. Convert numpy → Python string
            input_text = input_text.as_numpy()[0].decode('utf-8')
            
            # Get optional parameters
            try:
                max_tokens = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")
                max_tokens = int(max_tokens.as_numpy()[0])
            except:
                max_tokens = 200
            
            try:
                temperature = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
                temperature = float(temperature.as_numpy()[0])
            except:
                temperature = 0.8
            
            # Generate text (Model internally uses PyTorch tensors)
            full_prompt = self.system_prompt + "\\n\\n" + input_text
            
            # 3. Tokenizer converts string → PyTorch tensors internally
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # OUTPUT: Model → Triton (Tensor → Numpy conversion)
            # 4. Decode PyTorch tensor → Python string
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            output_text = output_text[len(full_prompt):].strip()
            
            # Free GPU memory after generation
            torch.cuda.empty_cache()
            
            # 5. Convert Python string → numpy array for Triton
            output_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                np.array([output_text.encode('utf-8')], dtype=np.object_)
            )
            
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        
        return responses
    
    def finalize(self):
        """Cleanup"""
        print("Cleaning up text generation model")
'''

CONFIG_PBTXT_TEMPLATE = '''name: "{model_name}"
backend: "python"

input [
  {{
    name: "INPUT_TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }},
  {{
    name: "MAX_TOKENS"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "TEMPERATURE"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "OUTPUT_TEXT"
    data_type: TYPE_STRING
    dims: [ -1 ]
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


def prepare_model(model_path: str, output_dir: str, model_name: str = "mistral_dm"):
    """Prepare text generation model repository for Triton"""
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        print(f"Error: Model path not found: {{model_path}}")
        exit(1)
    
    print(f"Preparing Triton model repository for text generation model...")
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
    model_py = MODEL_PY_TEMPLATE.format(system_prompt=DUNGEON_MASTER_PROMPT)
    (version_dir / "model.py").write_text(model_py)
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
    print(f"    └── [text generation model components]")

def main():
    parser = argparse.ArgumentParser(description="Prepare text generation model for Triton")
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
        default="mistral_dm",
        help="Model name in Triton (default: mistral_dm)"
    )
    
    args = parser.parse_args()
    
    prepare_model(args.model_path, args.output, args.model_name)


if __name__ == "__main__":
    main()

