#!/bin/bash
# AI Dungeon Master - Interactive Complete Setup
# This script installs every requirement and deploys the AI Dungeon Master project

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header "AI Dungeon Master - Complete Setup"

echo "This script will:"
echo "  • Install prerequisites (Docker, Python, tools)"
echo "  • Install NVIDIA drivers (if needed)"
echo "  • Setup MicroK8s with GPU support"
echo "  • Setup Python virtual environment"
echo "  • Download AI models"
echo "  • Deploy the application"
echo ""
echo "Estimated time: 30-60 minutes (depending on downloads)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

print_header "Step 1/7: Checking Prerequisites"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please run this script without sudo (it will prompt when needed)"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker installed: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"
    DOCKER_INSTALLED=true
else
    print_warning "Docker not installed"
    DOCKER_INSTALLED=false
fi

# Check Python
if command -v python3 &> /dev/null; then
    print_success "Python installed: $(python3 --version | cut -d' ' -f2)"
    PYTHON_INSTALLED=true
else
    print_warning "Python not installed"
    PYTHON_INSTALLED=false
fi

# Check NVIDIA drivers
if nvidia-smi &> /dev/null; then
    print_success "NVIDIA drivers installed"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_success "Detected: $GPU_COUNT x $GPU_NAME"
    NVIDIA_INSTALLED=true
else
    if command -v nvidia-smi &> /dev/null; then
        print_warning "NVIDIA drivers installed but not working (reboot may be required)"
    else
        print_warning "NVIDIA drivers not installed"
    fi
    NVIDIA_INSTALLED=false
fi

# ============================================================================
# Step 2: Install Prerequisites
# ============================================================================

if [ "$DOCKER_INSTALLED" = false ] || [ "$PYTHON_INSTALLED" = false ]; then
    print_header "Step 2/7: Installing Prerequisites"
    
    echo "Installing Docker, Python, and development tools..."
    "$PROJECT_DIR/scripts/install_prerequisites.sh"
    
    print_success "Prerequisites installed"
    
    # Check if we need to refresh groups
    if ! groups | grep -q docker; then
        print_warning "Docker group added. You need to refresh your shell."
        echo ""
        echo "Please run this command and then re-run this script:"
        echo "  newgrp docker"
        echo ""
        exit 0
    fi
else
    print_header "Step 2/7: Prerequisites Already Installed"
    print_success "Skipping prerequisite installation"
fi

# ============================================================================
# Step 3: NVIDIA Drivers
# ============================================================================

print_header "Step 3/7: NVIDIA Drivers"

if [ "$NVIDIA_INSTALLED" = false ]; then
    echo "NVIDIA drivers are required for GPU acceleration."
    echo ""
    read -p "Install NVIDIA drivers now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo "$PROJECT_DIR/scripts/install_nvidia_drivers.sh"
        
        print_warning "REBOOT REQUIRED!"
        echo ""
        echo "NVIDIA drivers installed successfully."
        echo "You must reboot before continuing."
        echo ""
        echo "After reboot, run this script again."
        echo ""
        read -p "Reboot now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo reboot
        fi
        exit 0
    else
        print_error "NVIDIA drivers are required. Exiting."
        exit 1
    fi
else
    if nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers already installed"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "GPU: $GPU_COUNT x $GPU_NAME"
    else
        print_error "NVIDIA drivers detected but not working!"
        echo ""
        echo "This usually means a reboot is required after driver installation."
        echo ""
        read -p "Reboot now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo reboot
        else
            print_error "Cannot continue without working GPU. Please reboot and run setup again."
            exit 1
        fi
    fi
fi

# ============================================================================
# Step 4: MicroK8s Setup
# ============================================================================

print_header "Step 4/7: Setting up MicroK8s"

if command -v microk8s &> /dev/null && microk8s status --wait-ready &> /dev/null; then
    print_success "MicroK8s already installed and running"
else
    "$PROJECT_DIR/scripts/setup_microk8s.sh"
    
    # Check if we need to refresh groups
    if ! groups | grep -q microk8s; then
        print_warning "MicroK8s group added. You need to refresh your shell."
        echo ""
        echo "Please run this command and then re-run this script:"
        echo "  newgrp microk8s"
        echo ""
        exit 0
    fi
fi

# Wait for GPU operator to be ready
echo "Waiting for GPU operator to be ready (this may take 1-2 minutes)..."
timeout=120
elapsed=0
while ! microk8s kubectl get pods -n gpu-operator-resources &> /dev/null; do
    if [ $elapsed -ge $timeout ]; then
        print_warning "GPU operator taking longer than expected, continuing anyway..."
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if microk8s kubectl get pods -n gpu-operator-resources &> /dev/null; then
    print_success "GPU operator deployed"
fi

# ============================================================================
# Step 5: GPU Configuration
# ============================================================================

print_header "Step 5/7: GPU Configuration"

# Check GPU status and count available GPUs
echo "Checking GPU status..."
echo "Your system has: $GPU_COUNT GPU(s)"
echo ""

GPU_FREE_COUNT=0
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)

# Show status for each GPU and count free ones
gpu_index=0
while IFS= read -r mem_used; do
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "$((gpu_index + 1))p")
    if [ "$mem_used" -gt 1000 ]; then
        echo "  GPU $gpu_index ($gpu_name): ${mem_used}MB used ⚠️  IN USE"
    else
        echo "  GPU $gpu_index ($gpu_name): ${mem_used}MB used ✓ Available"
        GPU_FREE_COUNT=$((GPU_FREE_COUNT + 1))
    fi
    gpu_index=$((gpu_index + 1))
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

if [ "$GPU_PROCESSES" -gt 0 ]; then
    echo ""
    echo "  Compute processes running: $GPU_PROCESSES"
fi

echo ""
echo "Available GPUs: $GPU_FREE_COUNT / $GPU_COUNT"
echo ""

# Check if we have any free GPUs
if [ "$GPU_FREE_COUNT" -eq 0 ]; then
    print_error "No GPUs available!"
    echo ""
    echo "All GPUs are currently in use. The AI Dungeon Master requires at least 1 free GPU."
    echo ""
    echo "Please free up at least one GPU by:"
    echo "  • Closing GPU-accelerated applications (browsers with hardware acceleration, games, etc.)"
    echo "  • Stopping other ML workloads"
    echo "  • Logging out of desktop environment (if using GUI)"
    echo ""
    echo "Then run this setup script again."
    exit 1
fi

# Configure deployment mode based on available GPUs
echo "GPU Configuration:"
echo ""

if [ "$GPU_FREE_COUNT" -eq 1 ]; then
    # Only 1 free GPU - auto-select single GPU mode
    GPU_MODE="single"
    print_success "1 free GPU available - using single GPU mode (text and image models will share GPU)"
else
    # 2+ free GPUs - ask user preference
    echo "You have $GPU_FREE_COUNT free GPUs available."
    echo ""
    echo "Deployment options:"
    echo "  1) Single GPU mode - Use 1 GPU for both models (~5-8s latency per request)"
    echo "  2) Dual GPU mode - Use 2 GPUs, one per model (~3-5s latency per request)"
    echo ""
    read -p "Choose (1 or 2, default=1): " gpu_choice
    
    if [ "$gpu_choice" = "2" ]; then
        GPU_MODE="dual"
        print_success "Dual GPU mode selected"
    else
        GPU_MODE="single"
        print_success "Single GPU mode selected"
    fi
fi

# Apply GPU operator fix for MicroK8s snap confinement
# WORKAROUND: MicroK8s snap confinement prevents the NVIDIA validator from creating
# symlinks in /dev/char for NVIDIA character devices (nvidiactl, nvidia0, etc).
# This is a known issue: https://github.com/NVIDIA/gpu-operator/issues/430
# The symlinks are only needed for systemd cgroup management in some configurations.
# With host-installed drivers in MicroK8s, they're not required, so we disable them.
echo ""
echo "Applying GPU operator fix for MicroK8s..."
echo "  (Disabling /dev/char symlink creation to work around snap confinement)"
echo "  See: https://github.com/NVIDIA/gpu-operator/issues/430"
echo "Waiting for GPU operator pods to be created..."
sleep 15

# Wait for daemonsets to exist
for i in {1..30}; do
    if microk8s kubectl get daemonset nvidia-container-toolkit-daemonset -n gpu-operator-resources &> /dev/null && \
       microk8s kubectl get daemonset nvidia-operator-validator -n gpu-operator-resources &> /dev/null; then
        break
    fi
    sleep 2
done

# Disable symlink creation on both validator daemonsets
microk8s kubectl set env daemonset/nvidia-container-toolkit-daemonset -n gpu-operator-resources \
    --containers=driver-validation DISABLE_DEV_CHAR_SYMLINK_CREATION=true 2>&1 || true

microk8s kubectl set env daemonset/nvidia-operator-validator -n gpu-operator-resources \
    --containers=driver-validation DISABLE_DEV_CHAR_SYMLINK_CREATION=true 2>&1 || true

print_success "GPU operator fix applied"

echo "Waiting for GPU operator to apply changes (30 seconds)..."
sleep 30

# ============================================================================
# Step 6: Python Environment and Models
# ============================================================================

print_header "Step 6/7: Python Environment and AI Models"

# Setup Python environment
if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
    # Remove broken venv directory if it exists
    if [ -d "$PROJECT_DIR/venv" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf "$PROJECT_DIR/venv"
    fi
    "$PROJECT_DIR/scripts/setup_python_env.sh"
else
    print_success "Python virtual environment already exists"
fi

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

# Verify required packages are installed
echo "Verifying Python packages..."
if ! python -c "import transformers, torch, diffusers" 2>/dev/null; then
    echo "Installing required packages for model download..."
    echo "This may take a few minutes (downloading PyTorch, Transformers, etc.)..."
    
    # Create temp dir on disk (not in tmpfs) for large pip downloads
    export TMPDIR="$HOME/.cache/pip-tmp"
    mkdir -p "$TMPDIR"
    
    # Install packages
    pip install -r "$PROJECT_DIR/models/requirements.txt"
    
    # Clean up temp directory
    rm -rf "$TMPDIR"
    unset TMPDIR
    
    print_success "Packages installed"
else
    print_success "Required packages already installed"
fi

# Check if models exist
TEXT_EXISTS=false
IMAGE_EXISTS=false

if [ -d "$PROJECT_DIR/models/text/triton_model_repo/mistral_dm" ]; then
    print_success "Text model already downloaded"
    TEXT_EXISTS=true
fi

if [ -d "$PROJECT_DIR/models/image/triton_model_repo/sdxl_pov" ]; then
    print_success "Image model already downloaded"
    IMAGE_EXISTS=true
fi

# Download models if needed
if [ "$TEXT_EXISTS" = false ] || [ "$IMAGE_EXISTS" = false ]; then
    echo ""
    print_header "AI Models Download"
    echo "The following models will be downloaded:"
    echo "  • Text model: Mistral 7B (~28 GB)"
    echo "  • Image model: Stable Diffusion XL (~72 GB)"
    echo ""
    echo "Total size: ~100 GB"
    echo ""
    
    read -p "Download models now? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Models are required. Exiting."
        echo ""
        echo "To download models manually, see README.md"
        exit 1
    fi
    
    # Download text model
    if [ "$TEXT_EXISTS" = false ]; then
        echo ""
        echo "Downloading text model (Mistral 7B, ~28 GB)..."
        cd "$PROJECT_DIR/models/text"
        python download_model.py --model mistralai/Mistral-7B-Instruct-v0.2 --output ./base_model || {
            print_error "Failed to download Mistral 7B"
            cd "$PROJECT_DIR"
            exit 1
        }
        print_success "Text model downloaded"
        
        echo "Preparing text model for Triton..."
        python prepare_triton_model.py --model-path ./base_model --output ./triton_model_repo || {
            print_error "Failed to prepare text model"
            cd "$PROJECT_DIR"
            exit 1
        }
        print_success "Text model prepared"
        
        # Remove empty base_model directory
        rm -rf ./base_model
        cd "$PROJECT_DIR"
    fi
    
    # Download image model
    if [ "$IMAGE_EXISTS" = false ]; then
        echo ""
        echo "Downloading image model (Stable Diffusion XL, ~72 GB)..."
        cd "$PROJECT_DIR/models/image"
        python download_model.py --model stabilityai/stable-diffusion-xl-base-1.0 --output ./base_model || {
            print_error "Failed to download Stable Diffusion XL"
            cd "$PROJECT_DIR"
            exit 1
        }
        print_success "Image model downloaded"
        
        echo "Preparing image model for Triton..."
        python prepare_triton_model.py --model-path ./base_model --output ./triton_model_repo || {
            print_error "Failed to prepare image model"
            cd "$PROJECT_DIR"
            exit 1
        }
        print_success "Image model prepared"
        
        # Remove empty base_model directory
        rm -rf ./base_model
        cd "$PROJECT_DIR"
    fi
fi

# ============================================================================
# Step 7: Deploy Application
# ============================================================================

print_header "Step 7/7: Deploying Application"

echo "Deploying AI Dungeon Master in $GPU_MODE GPU mode..."
echo ""

if [ "$GPU_MODE" = "single" ]; then
    "$PROJECT_DIR/scripts/deploy_single_gpu.sh"
else
    "$PROJECT_DIR/scripts/deploy_dual_gpu.sh"
fi

# ============================================================================
# Final Status
# ============================================================================

print_header "Setup Complete!"

echo "AI Dungeon Master is now deploying..."
echo ""
echo "Checking deployment status..."
sleep 5

# Check pod status
echo ""
echo "Deployment Status:"
microk8s kubectl get pods -n ai-dungeon-master

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Access your AI Dungeon Master:"
echo "  • Local:    http://localhost:30080 (NodePort)"
echo "  • External: http://<EXTERNAL-IP>   (Ingress on port 80)"
echo ""
echo "Useful commands:"
echo "  • Check status:  microk8s kubectl get pods -n ai-dungeon-master"
echo "  • View logs:     microk8s kubectl logs -n ai-dungeon-master -l app=fastapi --tail=100 -f"
echo "  • GPU usage:     watch nvidia-smi"
echo ""

