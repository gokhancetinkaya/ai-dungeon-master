# Custom Triton Server with Python ML dependencies
# Based on NVIDIA Triton Server with Python backend
# Using recent version from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

FROM nvcr.io/nvidia/tritonserver:25.01-py3

# Install Python packages required by the models
# These are needed for the Python backend to load text and image models
# Uninstall any existing torch first, then install latest compatible versions for CUDA 12.4
RUN pip uninstall -y torch torchvision torchaudio || true && \
    pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Install ML libraries (latest stable versions)
# transformers[vision] includes all dependencies for vision models
# scipy and scikit-image needed for diffusers
# numpy<2 required for tritonserver compatibility
RUN pip install --no-cache-dir \
    'numpy<2' \
    'transformers[vision]' \
    diffusers \
    accelerate \
    sentencepiece \
    protobuf \
    pillow \
    safetensors \
    timm \
    opencv-python-headless \
    scipy \
    scikit-image

# Set working directory
WORKDIR /models

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/live || exit 1

# The entrypoint is inherited from the base image (tritonserver)

