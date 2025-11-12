# AI Dungeon Master - Multi-Model Orchestration Platform

A complete demonstration platform for GPU-accelerated, multi-model AI orchestration. Features real-time narrative generation and scene rendering for D&D-style gameplay, deployed on Kubernetes with NVIDIA Triton Inference Server.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [GPU Setup Options](#gpu-setup-options)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Advanced Topics](#advanced-topics)
- [Resources](#resources)

---

## Overview

This project demonstrates:
- **Real-time narrative generation** using Mistral-7B-Instruct
- **Scene image generation** using Stable Diffusion XL
- **Multi-model coordination** via FastAPI orchestrator
- **GPU-accelerated inference** using NVIDIA Triton Inference Server
- **Container orchestration** on MicroK8s (Kubernetes)

### Demo Flow

```
User: "I open the ancient door"
   ↓
Text model generates narrative:
"The heavy oak door creaks open, revealing a dimly lit corridor. 
 Torches flicker on stone walls, casting dancing shadows.
 [SCENE: first-person view of dark medieval corridor, flickering torches]"
   ↓
FastAPI extracts scene description:
"first-person view of dark medieval corridor, flickering torches"
   ↓
Image model generates visual
   ↓
User sees: Narrative text + Scene image
```

**Total response time**: 3-8 seconds (depending on GPU setup)

---

## Architecture

### High-Level Overview
```
┌──────────────────────────────────────────┐
│        Frontend UI (Streamlit)           │
│  ┌──────────────┐    ┌──────────────┐    │
│  │  Chat Box    │    │Image Display │    │
│  └──────────────┘    └──────────────┘    │
└──────────────────┬───────────────────────┘
                   │ HTTP REST API
                   ▼
┌──────────────────────────────────────────┐
│         API Gateway (FastAPI)            │
│  • Routes requests to models             │
│  • Extracts scene descriptions           │
│  • Manages session state                 │
└────────────┬─────────────────┬───────────┘
             │  gRPC           │  gRPC
             ▼                 ▼
    ┌───────────────────────────────────┐
    │    Inference Server (Triton)      │
    │  ┌──────────┐    ┌──────────┐     │
    │  │   Text   │    │  Image   │     │
    │  │ (Mistral)│    │  (SDXL)  │     │
    │  └──────────┘    └──────────┘     │
    └─────────────────┬─────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │  Kubernetes Pods │
            │    (MicroK8s)    │
            │  + NVIDIA GPUs   │
            └──────────────────┘
```

### Deployment Modes

**Single GPU Mode** (1 GPU available):
```
┌─────────────────────────────────┐
│  Triton Server (Combined)       │
│  ┌─────────────────────────┐    │
│  │  GPU 0 (24GB VRAM)      │    │
│  │  • Mistral 7B           │    │
│  │  • SDXL                 │    │
│  │  Total: ~15GB used      │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
```

**Dual GPU Mode** (2+ GPUs available):
```
┌─────────────────┐     ┌─────────────────┐
│ Triton Server   │     │ Triton Server   │
│   (Text)        │     │   (Image)       │
│ ┌─────────────┐ │     │ ┌─────────────┐ │
│ │   GPU 0     │ │     │ │   GPU 1     │ │
│ │ Mistral 7B  │ │     │ │    SDXL     │ │
│ │   ~14GB     │ │     │ │    ~7GB     │ │
│ └─────────────┘ │     │ └─────────────┘ │
└─────────────────┘     └─────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| OS | Ubuntu | Base system |
| Orchestration | MicroK8s | Kubernetes |
| Model Serving | NVIDIA Triton | GPU inference (PyTorch backend) |
| API Gateway | FastAPI | Async orchestration |
| Frontend | Streamlit | User interface |
| Text Model | Mistral-7B-Instruct | Narrative generation |
| Image Model | Stable Diffusion XL | Scene rendering |

---

## Prerequisites

### Hardware Requirements
- **Ubuntu 24.04 LTS**
- **NVIDIA GPU(s)** with CUDA support
  - **Single GPU**: 1x 24GB+ VRAM (L4, RTX 3090/4090, A100)
  - **Dual GPU**: 2x 16GB+ VRAM each (better performance)
- **32GB+ RAM**
- **200GB+ storage** (for models)

### Software Requirements

**System:**
- **NVIDIA Driver**: 580.95.05
- **CUDA**: 13.0
- **Docker**: 28.5.1
- **MicroK8s**: 1.28.15
- **Python**: 3.12.3

**Python Dependencies (Model Preparation - venv):**
- PyTorch: 2.8.0
- Transformers: 4.57.1
- Diffusers: 0.35.2
- Accelerate: 1.11.0
- xformers: 0.0.32.post2
- Hugging Face Hub: 0.36.0
- NumPy: 2.3.4
- Pillow: 12.0.0
- tqdm: 4.67.1

**Python Dependencies (Runtime - Docker containers):**

*FastAPI Gateway:*
- FastAPI: 0.104.1
- Uvicorn: 0.24.0
- Triton Client: 2.40.0
- Pydantic: 2.5.0
- python-multipart: 0.0.6
- python-dotenv: 1.0.0
- aiohttp: 3.9.1
- websockets: 12.0
- NumPy: 1.24.3
- Pillow: 10.1.0

*Streamlit Frontend:*
- Streamlit: 1.29.0
- Requests: 2.31.0
- Pillow: 10.1.0

*Other versions may work but have not been tested.*

---

## Installation

### Quick Start (Recommended)

```bash
./setup.sh
```

This interactive script handles everything:
- Installs prerequisites (Docker, Python, tools)
- Installs NVIDIA drivers (prompts for reboot)
- Sets up MicroK8s with GPU support
- Configures GPU setup (asks for 1 or 2 GPUs)
- Downloads AI models (~100 GB)
- Deploys the application

### Manual Installation (Step-by-Step)

#### Step 1: Install Prerequisites

```bash
./scripts/install_prerequisites.sh
newgrp docker  # Refresh group after Docker install
```

#### Step 2: Install NVIDIA Drivers

```bash
sudo ./scripts/install_nvidia_drivers.sh
sudo reboot  # Required for driver installation

# After reboot, verify
nvidia-smi
```

#### Step 3: Setup MicroK8s

```bash
./scripts/setup_microk8s.sh
newgrp microk8s  # Refresh group membership

# Verify GPU detection
kubectl get nodes
kubectl describe node | grep -A 5 "Capacity"
```

#### Step 4: Setup Python Environment

```bash
./scripts/setup_python_env.sh
source venv/bin/activate

# Verify PyTorch
pip list | grep torch
```

#### Step 5: Download Models

##### Text Model: Mistral 7B Instruct

**Model Details:**
- Size: ~28GB download
- VRAM: ~14GB (FP16)
- No Hugging Face token required

**Download:**

```bash
cd models/text
python download_model.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --output ./base_model
```

##### Image Model: Stable Diffusion XL

**Model Details:**
- Size: ~72GB download
- VRAM: ~7GB (FP16, full model on GPU)
- No Hugging Face token required

**Download:**

```bash
cd models/image
python download_model.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --output ./base_model
```

#### Customize Models (Optional)

**⚠️ Do this BEFORE running the prepare scripts!**

##### Customize Dungeon Master Personality

Edit `models/text/prepare_triton_model.py` to change the DM's behavior:

```python
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
```

##### Customize Image Dimensions

Edit `models/image/prepare_triton_model.py` to change image size:

```python
# Image dimensions
height=768,  # Change to desired height
width=768    # Change to desired width
```

*Note: Image dimensions cannot be changed after deployment. For quality/guidance settings, see [Configuration](#configuration).*

##### Prepare Models for Triton

After downloading (and optionally customizing), prepare the models:

```bash
# Prepare text model
cd models/text
python prepare_triton_model.py \
    --model-path ./base_model \
    --output ./triton_model_repo \
    --model-name mistral_dm

# Prepare image model
cd ../image
python prepare_triton_model.py \
    --model-path ./base_model \
    --output ./triton_model_repo \
    --model-name sdxl_pov
```

#### Step 6: Deploy

Choose based on your GPU configuration (see [GPU Setup Options](#gpu-setup-options)):

**Single GPU**:
```bash
cd ../..
./scripts/deploy_single_gpu.sh
```

**Dual GPU**:
```bash
cd ../..
./scripts/deploy_dual_gpu.sh
```

#### Step 7: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n ai-dungeon-master
```

---

## GPU Setup Options

### Quick Comparison

| Option | GPUs | VRAM | Deployment | Latency | Best For |
|--------|------|------|------------|---------|----------|
| **Single GPU** | 1 | 24GB | `deploy_single_gpu.sh` | 5-8s | Demo, Development |
| **Dual GPU** | 2 | 2x16GB | `deploy_dual_gpu.sh` | 3-4s | Production, Events |

### Single GPU Setup

**Best for**: Cost-effective, simpler deployment, demo

#### Requirements
- 1x NVIDIA GPU with 24GB+ VRAM
  - RTX 3090 (24GB)
  - RTX 4090 (24GB)
  - A100 (40GB)

#### How It Works
- Single Triton server hosts both models on same GPU
- Text and Image models share VRAM
- Sequential processing (Text → Image)
- VRAM usage: ~15GB idle, ~18GB peak (24GB GPU recommended)

### Dual GPU Setup

**Best for**: Production, lowest latency, better scaling

#### Requirements
- 2x NVIDIA GPUs with 16GB+ VRAM each

#### How It Works
- Separate Triton servers for each model
- **GPU 0**: Text model (uses ~14GB of 16GB+ GPU)
- **GPU 1**: Image model (uses ~7GB of 16GB+ GPU)
- **Parallel processing**

### GPU Memory Requirements

**Measured VRAM Usage:**

**Single GPU (Combined)**:
- **Idle**: ~15GB (model weights loaded)
  - Mistral 7B: ~14GB (always on GPU)
  - SDXL: ~1GB (components in RAM, small overhead on GPU)
- **Peak**: ~18GB (during image generation with activations)
  - Mistral 7B: ~14GB (weights only, no activations)
  - SDXL: ~4GB (~3GB UNet weights + ~1GB activations on GPU)

**Dual GPU (Dedicated)**:
- **Idle**: ~21GB total (model weights loaded)
  - GPU 0: Mistral 7B: ~14GB
  - GPU 1: SDXL: ~7GB
- **Peak**: ~24-26GB total (during inference with activations)
  - GPU 0: Mistral 7B: ~16GB (14GB weights + ~2GB activations)
  - GPU 1: SDXL: ~8-10GB (7GB weights + ~1-3GB activations)

**What are activations?** Intermediate tensors created during inference (attention matrices, layer outputs, etc.). They're temporary and freed after each generation completes via `torch.cuda.empty_cache()`.

*Note: Combined deployment uses less total VRAM than the sum of individual models due to shared runtime overhead.*

### CPU Offloading Optimization

The image model (SDXL) uses conditional CPU offloading for optimal performance:

**Single GPU Mode**:
- `ENABLE_CPU_OFFLOAD=true`
- SDXL components (text encoder, UNet, VAE) stay in system RAM
- Only active component moves to GPU during inference
- Reduces SDXL GPU footprint from 7GB to 1-4GB peak
- Small latency penalty (~0.5s) but enables both models on one GPU

**Dual GPU Mode**:
- `ENABLE_CPU_OFFLOAD=false`
- Full SDXL model stays on GPU (7GB)
- No CPU↔GPU transfers, faster inference (~0.5s improvement)
- Better for production deployment with multiple GPUs

The environment variable is automatically set by the deployment scripts based on single vs. dual GPU mode.

---

## Usage

### Web Interface

**Access the application:**
- Via NodePort: **http://localhost:30080** (always works)
- Via Ingress: **http://localhost** (if ingress enabled)

**How to play:**
1. Open the URL in your browser
2. Type your action (e.g., "I enter the tavern")
3. Click "Take Action"
4. Watch the narrative appear on the left
5. See the scene image on the right

### Example Actions

- I enter the tavern
- I look around the room
- I approach the bartender
- I draw my sword
- I open the ancient door
- I search for traps
- I cast a light spell

---

## Configuration

**Runtime settings** can be changed after deployment without re-preparing models.

### Text Generation Settings

Edit `backend/fastapi_gateway/orchestrator.py` to adjust text generation:

```python
# In generate_narrative_and_image method
narrative = await self.llm_client.generate(
    prompt=prompt,
    max_tokens=200,        # Max response length (50-500)
    temperature=0.8        # Higher = more creative (0.0-2.0)
)
```

### Image Generation Settings

Edit `backend/fastapi_gateway/orchestrator.py` to adjust image quality:

```python
# In generate_narrative_and_image method
image_base64 = await self.image_client.generate(
    prompt=enhanced_prompt,
    num_inference_steps=20,  # Higher = better quality, slower (8-30)
    guidance_scale=7.5,      # How closely to follow prompt (1-20)
    negative_prompt=""       # What to avoid in images
)
```

### Applying Configuration Changes

After editing `orchestrator.py`, rebuild and redeploy FastAPI:

```bash
# 1. Rebuild FastAPI image
cd backend/fastapi_gateway
docker build -t ai-dungeon-master-fastapi:latest .
docker tag ai-dungeon-master-fastapi:latest localhost:32000/ai-dungeon-master-fastapi:latest
docker push localhost:32000/ai-dungeon-master-fastapi:latest

# 2. Restart FastAPI pods
kubectl rollout restart deployment fastapi -n ai-dungeon-master

# 3. Wait for rollout to complete
kubectl rollout status deployment fastapi -n ai-dungeon-master

# 4. Verify
kubectl get pods -n ai-dungeon-master -l app=fastapi
```

---

## Performance

### Expected Latency

| Setup | Text | Image | Total | Notes |
|-------|-----|-------|-------|-------|
| Single GPU (PyTorch) | 1-2s | 3-5s | **5-8s** | CPU offloading enabled |
| Dual GPU (PyTorch) | 1-2s | 1.5-2.5s | **3-4s** | No offloading, dedicated GPU |

### Resource Usage (Kubernetes Limits)

**Single GPU Deployment:**
| Service | CPU | RAM | GPU VRAM |
|---------|-----|-----|----------|
| Triton Combined | 3 cores | 24GB | 14.7GB (measured) |
| FastAPI | 1 core | 2GB | - |
| Frontend | 0.5 cores | 1GB | - |

**Dual GPU Deployment:**
| Service | CPU | RAM | GPU VRAM |
|---------|-----|-----|----------|
| Triton Text | 2 cores | 16GB | ~14GB |
| Triton Image | 2 cores | 8GB | ~7GB |
| FastAPI | 1 core | 2GB | - |
| Frontend | 0.5 cores | 1GB | - |

*Note: These are Kubernetes resource limits. Actual usage is typically lower.*

### Monitoring

```bash
# GPU usage
watch nvidia-smi

# Pod status
kubectl get pods -n ai-dungeon-master

# View logs (Single GPU setup):
kubectl logs -n ai-dungeon-master -l app=triton-combined --tail=100
kubectl logs -n ai-dungeon-master -l app=fastapi --tail=100

# View logs (Dual GPU setup):
kubectl logs -n ai-dungeon-master -l app=triton-text --tail=100    # Text model
kubectl logs -n ai-dungeon-master -l app=triton-image --tail=100   # Image model
kubectl logs -n ai-dungeon-master -l app=fastapi --tail=100

# Follow logs in real-time
kubectl logs -n ai-dungeon-master -l app=fastapi -f

# Filter out health check noise
kubectl logs -n ai-dungeon-master -l app=fastapi --tail=100 | grep -vE "GET /health|200 OK"

# Show only errors and important events
kubectl logs -n ai-dungeon-master -l app=fastapi --tail=100 | grep -E "ERROR|WARNING|INFO:main:|INFO:orchestrator"
```

---

## Project Structure

```
ai_dungeon_master/
├── README.md                    # This file
├── backend/
│   ├── fastapi_gateway/        # API orchestrator
│   │   ├── main.py             # FastAPI app + HTTP/WebSocket endpoints
│   │   ├── orchestrator.py     # Coordinates text + image model workflows
│   │   ├── triton_client.py    # gRPC clients for Triton Inference
│   │   ├── models.py           # Pydantic models (request/response validation)
│   │   ├── requirements.txt    # Python dependencies
│   │   └── Dockerfile          # Container image
│   └── triton/                 # Triton Inference Server config
├── frontend/
│   └── streamlit-ui/           # User interface
├── models/
│   ├── text/                   # Text model preparation scripts
│   └── image/                  # Image model scripts
├── kubernetes/                 # K8s deployment manifests
└── scripts/                    # Automation scripts
```

---

## API Reference

### Base URLs

**Via NodePort** (always works):
- Base URL: `http://localhost:30080`
- API Docs: `http://localhost:30080/docs`

**Via Ingress** (if ingress enabled):
- Base URL: `http://localhost`
- API Docs: `http://localhost/docs`

**Via Port Forward** (debugging):
```bash
kubectl port-forward -n ai-dungeon-master svc/fastapi-service 8000:8000
# Access: http://localhost:8000
```

### Endpoints

#### POST /api/chat

Send user message, get narrative + image response.

**Request:**
```bash
curl -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I enter the tavern",
    "session_id": "session-123"
  }'
```

**Response:**
```json
{
  "session_id": "session-123",
  "narrative": "You push through the heavy oak door...",
  "scene_description": "medieval tavern interior, warm firelight",
  "image_base64": "iVBORw0KGgoAAAANS...",
  "timestamp": "2025-11-01T12:00:00Z"
}
```

#### GET /health

Check service health and model availability.

```bash
curl http://localhost/health
```

**Response:**
```json
{
  "status": "healthy",
  "text_model_available": true,
  "image_model_available": true,
  "timestamp": "2025-11-01T12:00:00Z"
}
```

#### GET /api/models/status

Get detailed model status.

```bash
curl http://localhost/api/models/status
```

### Interactive Documentation

**OpenAPI/Swagger UI:**
- Via NodePort: http://localhost:30080/docs (always works)
- Via Ingress: http://localhost/docs (if ingress enabled)

**Features:**
- Try all endpoints interactively
- View request/response schemas
- Generate code samples

---

## Advanced Topics

### Production Deployment

For production:
1. Add authentication (API keys, OAuth)
2. Enable HTTPS/TLS
3. Setup monitoring (Prometheus, Grafana)
4. Add rate limiting
5. Use persistent storage (Redis for sessions)
6. Configure backups

---

## Resources

- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MicroK8s GPU Guide](https://microk8s.io/docs/addon-gpu)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Models](https://huggingface.co/models)

