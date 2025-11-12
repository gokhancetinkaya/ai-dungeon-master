#!/bin/bash
# Install NVIDIA drivers

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script with sudo"
    exit 1
fi

# Update system
echo "[1/4] Updating system..."
apt update && apt upgrade -y
apt install -y build-essential curl wget git vim ubuntu-drivers-common

# Check for NVIDIA GPU
echo "[2/4] Checking for NVIDIA GPU..."
if ! lspci | grep -i nvidia > /dev/null; then
    echo "ERROR: No NVIDIA GPU detected!"
    exit 1
fi
echo "✓ NVIDIA GPU detected"

# Install NVIDIA driver
echo "[3/3] Installing NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing recommended NVIDIA driver..."
    ubuntu-drivers install
    echo "✓ NVIDIA driver installed"
    echo "IMPORTANT: Reboot required! Run 'sudo reboot' after this script completes."
else
    echo "✓ NVIDIA driver already installed"
    nvidia-smi
fi

