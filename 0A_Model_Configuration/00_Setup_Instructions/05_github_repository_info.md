# GitHub Repository Information

## Primary Repository
- **URL:** https://github.com/CompVis/latent-diffusion
- **Clone Command:** `git clone https://github.com/CompVis/latent-diffusion.git`
- **Commit Hash:** [Add your specific commit hash here]
- **Branch:** main

## Important Forks
1. **Official CompVis:** https://github.com/CompVis/latent-diffusion
2. **Stable Diffusion (based on this):** https://github.com/Stability-AI/stablediffusion
3. **Community Forks:** Various forks with fixes and improvements

## Key Files in Repository

### Generation Scripts
- `scripts/txt2img.py` - Text to image generation
- `scripts/inpaint.py` - Image inpainting
- `scripts/sample_diffusion.py` - Sampling utilities

### Configuration Files
- `configs/latent-diffusion/` - Model configurations
- `configs/autoencoder/` - Autoencoder configurations
- `environment.yaml` - Conda environment

### Core Code
- `ldm/models/autoencoder.py` - Autoencoder implementation
- `ldm/models/diffusion/` - Diffusion model implementations
- `ldm/modules/` - Neural network modules

## Clone and Update Commands

```bash
# Clone repository
git clone https://github.com/CompVis/latent-diffusion.git

# Update to latest
cd latent-diffusion
git pull origin main

# Check specific commit (if needed)
git checkout [commit-hash]

# Create your own fork
# 1. Go to https://github.com/CompVis/latent-diffusion
# 2. Click "Fork" button
# 3. Clone your fork:
git clone https://github.com/YOUR_USERNAME/latent-diffusion.git
