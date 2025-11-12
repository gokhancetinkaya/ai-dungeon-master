# Custom Triton Server Image

This directory contains a custom Docker image for NVIDIA Triton Inference Server with Python ML dependencies.

## Why a Custom Image?

The base NVIDIA Triton Server image (`nvcr.io/nvidia/tritonserver:23.12-py3`) only includes the inference server itself. It does not include the Python packages required by our models:

- `torch` - Required by both text models (transformers) and image models (diffusers)
- `transformers` - Hugging Face library for text models (Mistral 7B)
- `diffusers` - Hugging Face library for image models (Stable Diffusion XL)
- `accelerate` - For efficient model loading
- `sentencepiece`, `protobuf` - Tokenizer dependencies
- `pillow`, `numpy`, `safetensors` - Image and data processing

## What's Included

The custom image extends the base Triton image and adds:

```
torch==2.1.0
transformers==4.35.0
diffusers==0.24.0
accelerate==0.25.0
sentencepiece==0.1.99
protobuf==3.20.3
pillow==10.1.0
numpy==1.24.3
safetensors==0.4.1
```

## Building

The image is automatically built by the deployment script:

```bash
./scripts/deploy_single_gpu.sh
```

Or manually:

```bash
cd backend/triton
docker build -t ai-dungeon-master-triton:latest .
docker tag ai-dungeon-master-triton:latest localhost:32000/ai-dungeon-master-triton:latest
docker push localhost:32000/ai-dungeon-master-triton:latest
```

## Size

- Base Triton image: ~7GB
- With Python packages: ~10-12GB (adds ~3-5GB)

The extra size is needed for PyTorch and the ML libraries, but this allows the models to load successfully in the container.

