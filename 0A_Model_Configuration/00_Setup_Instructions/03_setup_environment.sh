#!/bin/bash
# Environment setup for latent diffusion

echo "=== Environment Setup ==="

# Using Conda (recommended)
setup_conda() {
    echo "Setting up Conda environment..."
    
    # Check if conda exists
    if ! command -v conda &> /dev/null; then
        echo "Conda not found. Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b
        rm Miniconda3-latest-Linux-x86_64.sh
        source ~/.bashrc
    fi
    
    # Create environment
    conda env create -f environment.yaml
    conda activate ldm
    
    echo "Conda environment 'ldm' created and activated"
}

# Using pip only
setup_pip() {
    echo "Setting up with pip..."
    
    # Install PyTorch with CUDA 11.7
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install additional packages
    pip install omegaconf pytorch-lightning einops transformers \
        open-clip-torch pillow scikit-image scikit-learn
    
    echo "Pip setup complete"
}

# Check CUDA
check_cuda() {
    echo "Checking CUDA..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    else
        echo "No GPU detected. Using CPU-only setup."
    fi
    
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
}

# Main menu
echo "Select setup method:"
echo "1) Using Conda (recommended)"
echo "2) Using pip only"
echo "3) Check CUDA/GPU setup"
read -p "Choice [1-3]: " choice

case $choice in
    1) setup_conda ;;
    2) setup_pip ;;
    3) check_cuda ;;
    *) echo "Invalid choice" ;;
esac
