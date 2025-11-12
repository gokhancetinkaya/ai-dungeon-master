#!/bin/bash
# Setup Python virtual environment for model preparation

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_DIR="$PROJECT_DIR/venv"

# Check Python version
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Install python3-venv if needed
echo ""
echo "[2/5] Checking python3-venv..."
PYTHON_FULL_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
VENV_PACKAGE="python${PYTHON_FULL_VERSION}-venv"

if ! dpkg -l | grep -q "^ii.*$VENV_PACKAGE"; then
    echo "Installing $VENV_PACKAGE..."
    sudo apt install -y "$VENV_PACKAGE"
    echo "✓ $VENV_PACKAGE installed"
else
    echo "✓ $VENV_PACKAGE already installed"
fi

# Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "✓ Virtual environment recreated"
    else
        echo "Using existing virtual environment"
    fi
else
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
fi

# Activate and upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
source "$VENV_DIR/bin/activate"

# Use disk-based temp dir instead of tmpfs to avoid running out of space
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR"

pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "[5/5] Installing Python packages..."
if [ -f "$PROJECT_DIR/models/requirements.txt" ]; then
    echo "Installing model preparation dependencies..."
    echo "This may take a few minutes (downloading PyTorch, Transformers, etc.)..."
    pip install -r "$PROJECT_DIR/models/requirements.txt"
    echo "✓ Model dependencies installed"
fi

# Clean up temp directory
rm -rf "$TMPDIR"
unset TMPDIR

