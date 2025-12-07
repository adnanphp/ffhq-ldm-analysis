# ffhq_generate.py
import torch
import os
import sys

# Add the latent-diffusion directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import numpy as np

def analyze_checkpoint_parameters(checkpoint):
    """Analyze parameters in checkpoint"""
    print("\n" + "="*60)
    print("🔍 CHECKPOINT PARAMETER ANALYSIS")
    print("="*60)
    
    if "state_dict" in checkpoint:
        sd = checkpoint['state_dict']
        print(f" Found state_dict with {len(sd)} parameters")
    else:
        sd = checkpoint
        print(f" Using direct checkpoint with {len(sd)} parameters")
    
    # Extract parameter information
    param_counts = {}
    param_shapes = {}
    
    for key in sd.keys():
        # Extract parameter type from key
        parts = key.split('.')
        if len(parts) >= 2:
            param_type = parts[-2]
            param_counts[param_type] = param_counts.get(param_type, 0) + 1
            param_shapes[key] = sd[key].shape
    
    print("\n📊 Parameter Distribution:")
    for param_type, count in sorted(param_counts.items()):
        print(f"  {param_type}: {count} parameters")
    
    # Show critical parameters for understanding model structure
    print("\n CRITICAL PARAMETER KEYS:")
    critical_keys = [
        k for k in sd.keys() 
        if any(x in k for x in ['model.diffusion_model', 'first_stage_model', 'cond_stage_model'])
    ]
    
    for key in critical_keys[:15]:  # First 15 critical keys
        shape = sd[key].shape if hasattr(sd[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")
    
    # Analyze UNet structure from parameters
    unet_keys = [k for k in sd.keys() if 'model.diffusion_model' in k]
    print(f"\n UNet Analysis:")
    print(f"  Total UNet parameters: {len(unet_keys)}")
    
    # Get input block to understand channel structure
    for key in unet_keys:
        if 'input_blocks.0.0.weight' in key:
            shape = sd[key].shape
            print(f"  Input block: {shape[1]} -> {shape[0]} channels")
            break
    
    # Count attention layers
    attn_keys = [k for k in unet_keys if 'attn' in k]
    print(f"  Attention layers: {len(attn_keys)}")
    
    return sd

def main():
    print(" GENERATING FFHQ FACE")
    print("=" * 60)
    
    # 1. Create correct config
    config_yaml = """
model:
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    conditioning_key: crossattn
    scale_factor: 0.18215
    use_ema: false
    
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions: [8, 4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 3, 4]
        num_head_channels: 32
        
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        lossconfig:
          target: torch.nn.Identity
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        
    cond_stage_config:
      target: torch.nn.Identity
"""
    
    # Save config to file
    config_path = "ffhq_config_temp.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)
    
    print(f"✅ Created config file: {config_path}")
    
    # 2. Load config and analyze
    config = OmegaConf.load(config_path)
    
    print("\n CONFIGURATION ANALYSIS")
    print("-" * 40)
    
    if 'model' in config and 'params' in config.model:
        params = config.model.params
        
        print(" Key Config Parameters:")
        
        # UNet parameters
        if 'unet_config' in params and 'params' in params.unet_config:
            unet = params.unet_config.params
            print(f"\n  UNet Configuration:")
            print(f"    - model_channels: {unet.get('model_channels')} (CRITICAL: Must match checkpoint)")
            print(f"    - num_head_channels: {unet.get('num_head_channels')} (Instead of num_heads)")
            print(f"    - attention_resolutions: {unet.get('attention_resolutions')}")
            print(f"    - channel_mult: {unet.get('channel_mult')}")
        
        # VQGAN parameters
        if 'first_stage_config' in params and 'params' in params.first_stage_config:
            vqgan = params.first_stage_config.params
            print(f"\n  VQGAN Configuration:")
            print(f"    - embed_dim: {vqgan.get('embed_dim')}")
            print(f"    - n_embed: {vqgan.get('n_embed')}")
            if 'ddconfig' in vqgan:
                dd = vqgan['ddconfig']
                print(f"    - z_channels: {dd.get('z_channels')}")
                print(f"    - ch: {dd.get('ch')}")
        
        print(f"\n  Model Settings:")
        print(f"    - scale_factor: {params.get('scale_factor')} (IMPORTANT: 0.18215 vs 1.0)")
        print(f"    - conditioning_key: {params.get('conditioning_key')}")
    
    print("-" * 40)
    
    # 3. Create model
    print("\n Creating model from config...")
    model = instantiate_from_config(config.model)
    model.eval()
    model = model.cpu()
    
    print("✅ Model created")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Load checkpoint
    ckpt_path = "models/ldm/ffhq-ldm-vq-4/model.ckpt"
    print(f"\n Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Analyze checkpoint
    sd = analyze_checkpoint_parameters(checkpoint)
    
    print("\n Loading weights into model...")
    
    # Compare model and checkpoint parameters
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(sd.keys())
    
    print(f"\n PARAMETER MATCHING:")
    print(f"   Model keys: {len(model_keys)}")
    print(f"   Checkpoint keys: {len(checkpoint_keys)}")
    
    matching_keys = model_keys.intersection(checkpoint_keys)
    missing_keys = checkpoint_keys - model_keys
    extra_keys = model_keys - checkpoint_keys
    
    print(f"   Matching keys: {len(matching_keys)} ({len(matching_keys)/len(checkpoint_keys)*100:.1f}%)")
    
    if missing_keys:
        print(f"\n  Keys in checkpoint but NOT in model ({len(missing_keys)}):")
        for key in list(missing_keys)[:10]:
            print(f"   - {key}")
        if len(missing_keys) > 10:
            print(f"   ... and {len(missing_keys)-10} more")
    
    if extra_keys:
        print(f"\n  Keys in model but NOT in checkpoint ({len(extra_keys)}):")
        for key in list(extra_keys)[:10]:
            print(f"   + {key}")
        if len(extra_keys) > 10:
            print(f"   ... and {len(extra_keys)-10} more")
    
    # Load weights
    model.load_state_dict(sd, strict=False)
    print("\n✅ Checkpoint loaded")
    
    # 5. Create sampler
    sampler = DDIMSampler(model)
    
    # 6. Generate face
    print("\n" + "="*60)
    print(" GENERATING FACE...")
    print("=" * 60)
    
    # Best settings for FFHQ faces
    shape = [3, 64, 64]  # Latent shape
    c = torch.zeros(1, 0, 64, 64)  # Unconditional
    uc = torch.zeros(1, 0, 64, 64)
    
    print("\n⚙️  GENERATION SETTINGS:")
    print("  Steps: 150 (more steps = better quality)")
    print("  Guidance scale: 7.5 (optimal for faces)")
    print("  Sampler: DDIM (eta=0.0)")
    print("  Conditioning: Unconditional (zeros tensor)")
    print("  Latent shape: [3, 64, 64]")
    print("\n This will take a few minutes on CPU...")
    
    # Generate
    samples, _ = sampler.sample(
        S=150,  # Number of steps
        conditioning=c,
        batch_size=1,
        shape=shape,
        eta=0.0,  # DDIM
        verbose=False,
        unconditional_guidance_scale=7.5,  # Optimal for faces
        unconditional_conditioning=uc,
    )
    
    print(" Generation complete")
    
    # 7. Decode and save
    print("\n🔍 DECODING IMAGE...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
    
    # Convert to image
    img_np = x_samples[0].cpu().numpy().transpose(1, 2, 0)
    
    # Save
    output_path = "generated_face.png"
    Image.fromarray((img_np * 255).astype(np.uint8)).save(output_path)
    
    # Analyze image quality
    contrast = img_np.std()
    mean_brightness = img_np.mean()
    r, g, b = img_np.mean(axis=(0, 1))
    
    print(f"\n" + "="*60)
    print(" IMAGE ANALYSIS RESULTS:")
    print("=" * 60)
    print(f" Image saved: {output_path}")
    print(f"\n Image Quality Metrics:")
    print(f"  Contrast: {contrast:.3f} (good > 0.3)")
    print(f"  Brightness: {mean_brightness:.3f} (ideal 0.4-0.6)")
    print(f"  Colors - R: {r:.3f}, G: {g:.3f}, B: {b:.3f}")
    
    # Face likelihood check
    print(f"\n🔍 FACE LIKELIHOOD:")
    if contrast > 0.3:
        print(f"  ✅ HIGH CONTRAST: {contrast:.3f} (good for faces)")
        if r > g and g > b:
            print(f"  ✅ SKIN-LIKE COLORS: R > G > B")
            print(f"  🎉 VERY LIKELY A GOOD FACE IMAGE!")
        else:
            print(f"    Colors not skin-like, but structure may be good")
    elif contrast > 0.2:
        print(f"    MODERATE CONTRAST: {contrast:.3f}")
        print(f"   Check image - might be blurry face")
    else:
        print(f"  ❌ LOW CONTRAST: {contrast:.3f}")
        print(f"  🔍 Probably not a clear face")
    
    print(f"\n Open {output_path} to see the generated image!")
    
    # Key differences summary
    print("\n" + "="*60)
    print(" KEY DIFFERENCES FOR SUCCESS:")
    print("=" * 60)
    print("1. scale_factor: 0.18215 (vs 1.0 in failed attempt)")
    print("2. conditioning_key: crossattn (vs None in failed attempt)")
    print("3. num_head_channels: 32 (vs num_heads in failed attempt)")
    print("4. Guidance scale: 7.5 (vs 1.0 in failed attempt)")
    print("5. Steps: 150 (vs 50 in failed attempt)")
    print("6. Proper latent shape: [3, 64, 64]")
    
    # Cleanup
    os.remove(config_path)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
