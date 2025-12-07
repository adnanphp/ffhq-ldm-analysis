#!/bin/bash
# Verify the complete setup

echo "=== Setup Verification ==="

check_structure() {
    echo "1. Checking directory structure..."
    
    dirs=("latent-diffusion" "models" "models/ldm" "models/ldm/ffhq-ldm-vq-4")
    
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "    $dir"
        else
            echo "    $dir - Missing"
        fi
    done
}

check_files() {
    echo "2. Checking essential files..."
    
    files=(
        "latent-diffusion/scripts/txt2img.py"
        "models/ldm/ffhq-ldm-vq-4/model.ckpt"
        "latent-diffusion/configs/latent-diffusion/ffhq-ldm-vq-4.yaml"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "    $file ($size)"
        else
            echo "    $file - Missing"
        fi
    done
}

check_python() {
    echo "3. Checking Python packages..."
    
    python3 -c "
import importlib

packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('omegaconf', 'OmegaConf'),
    ('pytorch_lightning', 'PyTorch Lightning'),
    ('einops', 'Einops')
]

for module_name, display_name in packages:
    try:
        importlib.import_module(module_name)
        version = 'unknown'
        if module_name == 'torch':
            import torch; version = torch.__version__
        elif module_name == 'omegaconf':
            import omegaconf; version = omegaconf.__version__
        print(f'   ✅ {display_name}: {version}')
    except ImportError:
        print(f'   ❌ {display_name}: Not installed')
"
}

check_model() {
    echo "4. Checking model integrity..."
    
    if [ -f "models/ldm/ffhq-ldm-vq-4/model.ckpt" ]; then
        size=$(du -h "models/ldm/ffhq-ldm-vq-4/model.ckpt" | cut -f1)
        echo "    FFHQ model present ($size)"
        
        # Try to load with Python
        python3 -c "
import torch
try:
    ckpt = torch.load('models/ldm/ffhq-ldm-vq-4/model.ckpt', map_location='cpu')
    if 'state_dict' in ckpt:
        params = sum(p.numel() for p in ckpt['state_dict'].values())
    else:
        params = sum(p.numel() for p in ckpt.values())
    print(f'      Model has {params:,} parameters')
except Exception as e:
    print(f'      Error loading model: {e}')
"
    else
        echo "    FFHQ model missing"
    fi
}

# Run all checks
check_structure
check_files
check_python
check_model

echo ""
echo "=== Summary ==="
echo "Run test generation with:"
echo "  cd latent-diffusion && python scripts/txt2img.py --prompt 'test'"
echo ""
echo "If any checks failed, run:"
echo "  ./01_clone_repository.sh"
echo "  ./02_download_models.sh"
echo "  ./03_setup_environment.sh"
