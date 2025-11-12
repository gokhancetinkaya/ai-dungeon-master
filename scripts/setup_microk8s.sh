#!/bin/bash
# Setup MicroK8s for AI Dungeon Master

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please run this script without sudo (it will prompt when needed)"
    exit 1
fi

# Fix for Google Cloud and other platforms with long hostnames
# MicroK8s certificate generation fails if hostname is too long (>64 chars in CN field)
# Cloud VMs often have auto-generated hostnames like: instance-name.region.project.internal
echo "[1/8] Checking hostname for MicroK8s certificate compatibility..."
CURRENT_HOSTNAME=$(hostname)
HOSTNAME_LENGTH=${#CURRENT_HOSTNAME}

if [ $HOSTNAME_LENGTH -gt 50 ]; then
    echo "⚠ Hostname too long for Kubernetes certificates ($HOSTNAME_LENGTH chars)"
    echo "  Current: $CURRENT_HOSTNAME"
    echo "  MicroK8s SSL certificates require hostname ≤64 chars (you have $HOSTNAME_LENGTH)"
    echo ""
    
    # Generate a shorter hostname
    SHORT_HOSTNAME="ai-dungeon-master"
    
    echo "  Changing hostname to: $SHORT_HOSTNAME"
    sudo hostnamectl set-hostname "$SHORT_HOSTNAME"
    
    # Update /etc/hosts to include both names
    if ! grep -q "127.0.1.1.*$SHORT_HOSTNAME" /etc/hosts; then
        sudo sed -i "/^127.0.1.1/c\127.0.1.1 $SHORT_HOSTNAME $CURRENT_HOSTNAME" /etc/hosts
    fi
    
    echo "✓ Hostname changed to: $(hostname) (to fix certificate generation)"
else
    echo "✓ Hostname is compatible: $CURRENT_HOSTNAME ($HOSTNAME_LENGTH chars)"
fi

# Install MicroK8s
echo "[2/8] Installing MicroK8s..."
if ! command -v microk8s &> /dev/null; then
    sudo snap install microk8s --classic --channel=1.28/stable
    echo "✓ MicroK8s installed"
else
    echo "✓ MicroK8s already installed"
fi

# Add user to microk8s group
echo "[3/8] Configuring user permissions..."
sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube || true
echo "✓ User added to microk8s group"

# Wait for MicroK8s to be ready
echo "[4/8] Waiting for MicroK8s to be ready..."
sudo microk8s status --wait-ready
echo "✓ MicroK8s is ready"

# Enable required addons
echo "[5/8] Enabling MicroK8s addons..."

# DNS
if ! sudo microk8s status | grep -q "dns: enabled"; then
    sudo microk8s enable dns
    echo "✓ DNS enabled"
else
    echo "✓ DNS already enabled"
fi

# Storage (hostpath-storage)
if ! sudo microk8s status | grep -q "hostpath-storage: enabled"; then
    sudo microk8s enable hostpath-storage
    echo "✓ Storage enabled"
else
    echo "✓ Storage already enabled"
fi

# Registry (for local image storage)
if ! sudo microk8s status | grep -q "registry: enabled"; then
    sudo microk8s enable registry
    echo "✓ Registry enabled"
else
    echo "✓ Registry already enabled"
fi

# GPU support
echo "[6/8] Enabling GPU support..."
if ! sudo microk8s status | grep -q "gpu: enabled"; then
    sudo microk8s enable gpu
    echo "✓ GPU support enabled"
else
    echo "✓ GPU already enabled"
fi

# Ingress for external HTTP access
read -p "Enable ingress for external HTTP access? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[7/8] Enabling Ingress..."
    if ! sudo microk8s status | grep -q "ingress: enabled"; then
        sudo microk8s enable ingress
        echo "✓ Ingress enabled"
    else
        echo "✓ Ingress already enabled"
    fi
fi

# Setup kubectl alias
echo "[8/8] Setting up kubectl alias..."
if ! grep -q "alias kubectl='microk8s kubectl'" ~/.bashrc; then
    echo "alias kubectl='microk8s kubectl'" >> ~/.bashrc
    echo "✓ kubectl alias added to ~/.bashrc"
fi

