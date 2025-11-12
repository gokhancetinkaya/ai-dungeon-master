#!/bin/bash
# Install all prerequisites for AI Dungeon Master

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please run this script without sudo (it will prompt when needed)"
    exit 1
fi

# Update system
echo "[1/5] Updating system..."
sudo apt update
echo "✓ System updated"

# Install Docker
echo ""
echo "[2/5] Installing Docker..."
if command -v docker &> /dev/null; then
    echo "✓ Docker already installed ($(docker --version))"
else
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    echo "✓ Docker installed"
fi

# Install Python and venv
echo ""
echo "[3/5] Installing Python..."
if command -v python3 &> /dev/null; then
    echo "✓ Python already installed ($(python3 --version))"
else
    sudo apt install -y python3 python3-venv python3-pip
    echo "✓ Python installed"
fi

# Install additional tools
echo ""
echo "[4/4] Installing additional tools..."
sudo apt install -y curl wget git vim build-essential
echo "✓ Additional tools installed"

