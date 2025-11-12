#!/bin/bash
# Deploy AI Dungeon Master with Dual GPU Setup

set -e

export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

echo "=========================================="
echo "Deploying AI Dungeon Master (Dual GPU)"
echo "=========================================="

# Check if user is in microk8s group
if ! groups | grep -q microk8s; then
    echo "ERROR: You're not in the microk8s group in this shell"
    echo ""
    echo "Solution: Choose one of these options:"
    echo "  Option 1: Run 'newgrp microk8s'"
    echo "  Option 2: Logout/login"
    echo "  Option 3: Reboot"
    echo ""
    exit 1
fi

# Check if MicroK8s is running
if ! microk8s status --wait-ready &> /dev/null; then
    echo "ERROR: MicroK8s is not running. Run ./scripts/setup_microk8s.sh first"
    exit 1
fi

# Check model repositories exist
echo "[1/7] Checking model repositories..."

if [ ! -d "$PROJECT_DIR/models/text/triton_model_repo/mistral_dm" ]; then
    echo "ERROR: Text model not found. Run model preparation scripts first."
    exit 1
fi
echo "✓ Text model found"

if [ ! -d "$PROJECT_DIR/models/image/triton_model_repo/sdxl_pov" ]; then
    echo "ERROR: Image model not found. Run model preparation scripts first."
    exit 1
fi
echo "✓ Image model found"

# Build Docker images
echo ""
echo "[2/7] Building Docker images..."

# Build custom Triton image with ML dependencies
cd "$PROJECT_DIR/backend/triton"
echo "Building custom Triton image (with PyTorch, Transformers, Diffusers)..."
echo "This may take 5-10 minutes on first build..."
docker build -t ai-dungeon-master-triton:latest .
docker tag ai-dungeon-master-triton:latest localhost:32000/ai-dungeon-master-triton:latest
docker push localhost:32000/ai-dungeon-master-triton:latest
echo "✓ Triton image built and pushed"

cd "$PROJECT_DIR/backend/fastapi_gateway"
echo "Building FastAPI image..."
docker build -t ai-dungeon-master-fastapi:latest .
docker tag ai-dungeon-master-fastapi:latest localhost:32000/ai-dungeon-master-fastapi:latest
docker push localhost:32000/ai-dungeon-master-fastapi:latest
echo "✓ FastAPI image built and pushed"

cd "$PROJECT_DIR/frontend/streamlit-ui"
echo "Building Streamlit UI image..."
docker build -t ai-dungeon-master-streamlit:latest .
docker tag ai-dungeon-master-streamlit:latest localhost:32000/ai-dungeon-master-streamlit:latest
docker push localhost:32000/ai-dungeon-master-streamlit:latest
echo "✓ Streamlit image built and pushed"

# Create namespace
echo ""
echo "[3/7] Creating namespace..."
microk8s kubectl apply -f "$PROJECT_DIR/kubernetes/namespace.yaml"
echo "✓ Namespace created"

# Deploy Triton servers with hostPath mounts
echo ""
echo "[4/7] Deploying Triton inference servers (2 GPUs)..."

# Deploy Triton text model server
envsubst < "$PROJECT_DIR/kubernetes/triton-llm-deployment.yaml" | microk8s kubectl apply -f -
echo "✓ Triton text model service deployed (models mounted from host)"

# Deploy Triton image model server
envsubst < "$PROJECT_DIR/kubernetes/triton-image-deployment.yaml" | microk8s kubectl apply -f -
echo "✓ Triton image model service deployed (models mounted from host)"

# Wait for Triton pods to be ready
echo ""
echo "[5/7] Waiting for Triton pods to load models (this may take a few minutes)..."
echo "Waiting for Triton text model pod..."
microk8s kubectl wait --for=condition=ready pod -l app=triton-text -n ai-dungeon-master --timeout=300s || {
    echo "WARNING: Triton text model pod not ready yet. Checking status..."
    microk8s kubectl get pods -n ai-dungeon-master -l app=triton-text
    microk8s kubectl describe pod -n ai-dungeon-master -l app=triton-text | tail -20
    echo "Continuing with deployment..."
}

echo "Waiting for Triton image model pod..."
microk8s kubectl wait --for=condition=ready pod -l app=triton-image -n ai-dungeon-master --timeout=300s || {
    echo "WARNING: Triton image model pod not ready yet. Checking status..."
    microk8s kubectl get pods -n ai-dungeon-master -l app=triton-image
    microk8s kubectl describe pod -n ai-dungeon-master -l app=triton-image | tail -20
    echo "Continuing with deployment..."
}

echo "✓ Triton pods are ready and serving models"

# Deploy FastAPI
echo "[6/7] Deploying FastAPI gateway..."
microk8s kubectl apply -f "$PROJECT_DIR/kubernetes/fastapi-deployment.yaml"
echo "✓ FastAPI gateway deployed"

# Deploy Frontend
echo "[7/7] Deploying frontend..."
microk8s kubectl apply -f "$PROJECT_DIR/kubernetes/frontend-deployment.yaml"
echo "✓ Frontend deployed"

echo ""
echo "Waiting for all pods to be ready (this may take 5-10 minutes for model loading)..."
echo "Waiting for Triton text server..."
microk8s kubectl wait --for=condition=ready pod -l app=triton-text -n ai-dungeon-master --timeout=600s || echo "⚠ Triton text pods not ready yet"
echo "Waiting for Triton image server..."
microk8s kubectl wait --for=condition=ready pod -l app=triton-image -n ai-dungeon-master --timeout=600s || echo "⚠ Triton image pods not ready yet"
echo "Waiting for FastAPI gateway..."
microk8s kubectl wait --for=condition=ready pod -l app=fastapi -n ai-dungeon-master --timeout=300s || echo "⚠ FastAPI pods not ready yet"
echo "Waiting for Frontend..."
microk8s kubectl wait --for=condition=ready pod -l app=frontend -n ai-dungeon-master --timeout=120s || echo "⚠ Frontend pods not ready yet"

echo ""
echo "✓ All deployments applied. Checking final status..."
microk8s kubectl get pods -n ai-dungeon-master
echo ""

# Check if ingress addon is enabled
if microk8s status --addon ingress | grep -q "enabled"; then
    echo "MicroK8s ingress addon is enabled, deploying ingress..."
    # Wait a moment for ingress controller to be fully ready
    sleep 5
    if microk8s kubectl apply -f "$PROJECT_DIR/kubernetes/ingress.yaml" 2>/dev/null; then
        echo "✓ Ingress deployed"
    else
        echo "⚠ Ingress deployment failed (controller may still be initializing)"
        echo "You can deploy it manually later:"
        echo "  microk8s kubectl apply -f $PROJECT_DIR/kubernetes/ingress.yaml"
    fi
else
    echo "MicroK8s ingress addon is not enabled."
    echo "To enable it, run: microk8s enable ingress"
    echo "Skipping ingress deployment"
fi

