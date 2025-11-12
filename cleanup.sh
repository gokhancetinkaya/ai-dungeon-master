#!/bin/bash
# AI Dungeon Master - Cleanup
# This script removes all Kubernetes resources and model repositories for the AI Dungeon Master project

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

print_header "AI Dungeon Master - Kubernetes Cleanup"

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
    print_error "MicroK8s is not running"
    exit 1
fi

# Ask about MicroK8s removal first (since it removes everything)
echo "Do you want to:"
echo "  1) Clean up AI Dungeon Master resources (keep MicroK8s)"
echo "  2) Remove MicroK8s completely (removes everything)"
echo ""
read -p "Choose (1 or 2, default=1): " removal_choice
echo ""

if [ "$removal_choice" = "2" ]; then
    print_header "Removing MicroK8s"
    
    echo "This will completely remove MicroK8s and ALL Kubernetes resources."
    echo "You'll need to re-install it with ./setup.sh or manually."
    echo ""
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Stopping MicroK8s..."
        microk8s stop 2>&1 || true
        sleep 5
        
        echo "Removing MicroK8s snap..."
        sudo snap remove microk8s --purge 2>&1
        
        print_success "MicroK8s removed"
        
        echo ""
        echo "Cleaning up remaining directories..."
        sudo rm -rf /var/snap/microk8s 2>/dev/null || true
        sudo rm -rf ~/.kube 2>/dev/null || true
        
        print_success "MicroK8s completely removed"
        
        echo ""
        print_header "Complete Cleanup Done!"
        echo ""
        echo "To reinstall everything, run: ./setup.sh"
        echo ""
        exit 0
    else
        print_warning "MicroK8s removal cancelled, continuing with resource cleanup..."
        echo ""
    fi
fi

print_header "Cleaning up AI Dungeon Master Resources"

# Check if namespace exists
AI_DUNGEON_MASTER_EXISTS=false
if microk8s kubectl get namespace ai-dungeon-master &> /dev/null; then
    AI_DUNGEON_MASTER_EXISTS=true
    
    # Show current resources
    echo "Current resources in ai-dungeon-master namespace:"
    echo ""
    echo "Deployments:"
    microk8s kubectl get deployments -n ai-dungeon-master 2>/dev/null || echo "  None"
    echo ""
    echo "Services:"
    microk8s kubectl get services -n ai-dungeon-master 2>/dev/null || echo "  None"
    echo ""
    echo "Pods:"
    microk8s kubectl get pods -n ai-dungeon-master 2>/dev/null || echo "  None"
    echo ""
else
    print_warning "Namespace 'ai-dungeon-master' not found - skipping application cleanup"
    echo ""
fi

if [ "$AI_DUNGEON_MASTER_EXISTS" = true ]; then
    # Ask for confirmation
    read -p "Do you want to delete AI Dungeon Master resources? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        print_header "Deleting AI Dungeon Master namespace..."
        
        # Delete the namespace (this removes everything automatically)
        if microk8s kubectl delete namespace ai-dungeon-master; then
            print_success "AI Dungeon Master namespace deleted"
            print_success "All resources (pods, services, deployments, etc.) removed"
        else
            print_error "Failed to delete namespace"
        fi
    else
        print_warning "AI Dungeon Master cleanup skipped"
    fi
fi

# Check for GPU operator namespace
echo ""
if microk8s kubectl get namespace gpu-operator-resources &> /dev/null; then
    read -p "Delete GPU operator namespace 'gpu-operator-resources'? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Deleting GPU operator namespace..."
        echo "Note: This will disable GPU support."
        echo "Re-run ./setup.sh to configure GPU again."
        echo ""
        read -p "Are you sure? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            microk8s disable gpu 2>&1
            print_success "GPU operator disabled"
        else
            print_warning "GPU operator namespace preserved"
        fi
    else
        print_warning "GPU operator namespace preserved"
    fi
fi

# Clean up model repositories
echo ""
read -p "Clean up model repositories? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clean up text model directories
    if [ -d "$PROJECT_DIR/models/text/base_model" ]; then
        rm -rf "$PROJECT_DIR/models/text/base_model"
        print_success "Text model base files removed"
    fi
    if [ -d "$PROJECT_DIR/models/text/triton_model_repo" ]; then
        rm -rf "$PROJECT_DIR/models/text/triton_model_repo"
        print_success "Text model Triton repository removed"
    fi
    
    # Clean up image model directories
    if [ -d "$PROJECT_DIR/models/image/base_model" ]; then
        rm -rf "$PROJECT_DIR/models/image/base_model"
        print_success "Image model base files removed"
    fi
    if [ -d "$PROJECT_DIR/models/image/triton_model_repo" ]; then
        rm -rf "$PROJECT_DIR/models/image/triton_model_repo"
        print_success "Image model Triton repository removed"
    fi
else
    print_warning "Model repositories preserved"
fi

# Summary
echo ""
print_header "Cleanup Complete!"
echo ""
echo "Kubernetes resources have been cleaned up."
echo ""
echo "To redeploy:"
echo "  Run setup:           ./setup.sh"
echo "  Or manually:"
echo "    Single GPU:        ./scripts/deploy_single_gpu.sh"
echo "    Dual GPU:          ./scripts/deploy_dual_gpu.sh"
echo ""
echo "Check remaining resources:"
echo "  microk8s kubectl get all -n ai-dungeon-master"
echo ""

