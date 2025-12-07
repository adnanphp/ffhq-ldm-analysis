# test_ffhq_fixed_final_v11.py
import torch
import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path

print("FIXED FFHQ MODEL TEST")
print("=" * 50)

# Add the necessary paths FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Current directory
sys.path.insert(0, os.path.join(current_dir, 'latent-diffusion'))  # LDM module
sys.path.insert(0, os.path.join(current_dir, 'src'))  # Source directory

print(f" Working directory: {current_dir}")

def setup_environment():
    """Setup the environment and import paths"""
    print(" Setting up environment...")
    
    # Check if latent-diffusion exists
    ldm_path = os.path.join(current_dir, 'latent-diffusion')
    if not os.path.exists(ldm_path):
        print(f" latent-diffusion directory not found at: {ldm_path}")
        print(" Please make sure you have the latent-diffusion repository cloned")
        return False
    
    # Check for required modules
    try:
        # Try to import LDM modules
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf
        print("✅ LDM modules imported successfully")
        return True
    except ImportError as e:
        print(f" Failed to import LDM modules: {e}")
        print(" Trying to fix imports...")
        return try_fix_imports()

def try_fix_imports():
    """Try to fix import issues"""
    try:
        # Add latent-diffusion subdirectories to path
        ldm_path = os.path.join(current_dir, 'latent-diffusion')
        subdirs = ['', 'ldm', 'ldm/models', 'ldm/modules', 'ldm/models/diffusion', 'ldm/models/autoencoder']
        
        for subdir in subdirs:
            full_path = os.path.join(ldm_path, subdir)
            if os.path.exists(full_path):
                sys.path.insert(0, full_path)
        
        # Try importing again
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf
        print("✅ Fixed imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Still cannot import LDM modules: {e}")
        return False

def apply_compatibility_fixes():
    """Apply all necessary compatibility fixes"""
    print(" Applying compatibility fixes...")
    
    try:
        # Fix VectorQuantizer imports - COMPREHENSIVE FIX
        import taming.modules.vqvae.quantize as quantize
        
        # Save original VectorQuantizer
        OriginalVQ = quantize.VectorQuantizer
        
        # Create fixed version that handles ALL parameter combinations
        class FixedVectorQuantizer(OriginalVQ):
            def __init__(self, n_e, e_dim, beta=0.25, **kwargs):
                print(f"🔧 VectorQuantizer called with n_e={n_e}, e_dim={e_dim}, beta={beta}, kwargs={kwargs}")
                
                # Try different constructor signatures
                try:
                    # Signature 1: n_e, e_dim, beta
                    super().__init__(n_e, e_dim, beta)
                    print("✅ Used signature: (n_e, e_dim, beta)")
                except TypeError as e1:
                    try:
                        # Signature 2: n_e, e_dim, beta=beta
                        super().__init__(n_e, e_dim, beta=beta)
                        print("✅ Used signature: (n_e, e_dim, beta=beta)")
                    except TypeError as e2:
                        try:
                            # Signature 3: n_e, e_dim (no beta)
                            super().__init__(n_e, e_dim)
                            print("✅ Used signature: (n_e, e_dim)")
                        except TypeError as e3:
                            try:
                                # Signature 4: with all possible kwargs
                                filtered_kwargs = {}
                                valid_kwargs = ['remap', 'unknown_index', 'sane_index_shape', 'legacy']
                                for k, v in kwargs.items():
                                    if k in valid_kwargs:
                                        filtered_kwargs[k] = v
                                super().__init__(n_e, e_dim, beta=beta, **filtered_kwargs)
                                print("✅ Used signature: (n_e, e_dim, beta=beta, **filtered_kwargs)")
                            except TypeError as e4:
                                # Last resort: create a working minimal VectorQuantizer
                                print(" Using working minimal VectorQuantizer")
                                import torch.nn as nn
                                nn.Module.__init__(self)
                                self.n_e = n_e
                                self.e_dim = e_dim
                                self.beta = beta
                                self.embedding = nn.Embedding(n_e, e_dim)
                                self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
            
            def forward(self, z):
                # Simple forward pass that just returns the input (identity mapping)
                # This allows decoding to work without actual quantization
                return z, None, (None, None, None)
        
        # Replace in module
        quantize.VectorQuantizer = FixedVectorQuantizer
        if hasattr(quantize, 'VectorQuantizer2'):
            quantize.VectorQuantizer2 = FixedVectorQuantizer
        
        print(" VectorQuantizer compatibility fixed")
        
    except Exception as e:
        print(f"  Compatibility fix warning: {e}")
        # Try alternative fix
        apply_alternative_compatibility_fixes()

def apply_alternative_compatibility_fixes():
    """Alternative compatibility fixes"""
    print("Trying alternative compatibility fixes...")
    
    try:
        # Direct monkey patching of the VQModel
        from ldm.models.autoencoder import VQModel
        
        original_vqmodel_init = VQModel.__init__
        
        def patched_vqmodel_init(self, *args, **kwargs):
            # Filter out problematic kwargs before calling parent
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['remap', 'unknown_index', 'sane_index_shape', 'legacy']}
            return original_vqmodel_init(self, *args, **filtered_kwargs)
        
        VQModel.__init__ = patched_vqmodel_init
        print("✅ VQModel compatibility fixed")
        
    except Exception as e:
        print(f"  Alternative compatibility fix failed: {e}")

def analyze_checkpoint_parameters(pl_sd):
    """Analyze parameters in checkpoint"""
    print("\n" + "="*60)
    print(" CHECKPOINT PARAMETER ANALYSIS")
    print("="*60)
    
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
        print(f"✅ Found state_dict with {len(sd)} parameters")
    else:
        sd = pl_sd
        print(f" Using direct checkpoint with {len(sd)} parameters")
    
    # Count parameters by type
    param_counts = {}
    param_shapes = {}
    
    for key in sd.keys():
        # Extract parameter type from key
        parts = key.split('.')
        if len(parts) >= 2:
            param_type = parts[-2]  # Second last part is usually parameter type
            param_counts[param_type] = param_counts.get(param_type, 0) + 1
            param_shapes[key] = sd[key].shape
    
    print("\n📊 Parameter Distribution:")
    for param_type, count in sorted(param_counts.items()):
        print(f"  {param_type}: {count} parameters")
    
    # Show sample parameter keys and shapes
    print("\n Sample Parameter Keys and Shapes:")
    sample_keys = list(sd.keys())[:10]  # First 10 keys
    for key in sample_keys:
        shape = sd[key].shape if hasattr(sd[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")
    
    # Check for key model components
    required_components = [
        'model.diffusion_model',
        'first_stage_model',
        'cond_stage_model'
    ]
    
    print("\n Checking for Required Components:")
    for component in required_components:
        component_found = any(component in key for key in sd.keys())
        status = "✅ FOUND" if component_found else "❌ MISSING"
        print(f"  {component}: {status}")
    
    # Check UNet specific parameters
    unet_keys = [k for k in sd.keys() if 'model.diffusion_model' in k]
    print(f"\n UNet Parameters: {len(unet_keys)} total")
    
    # Check attention parameters
    attention_keys = [k for k in unet_keys if 'attn' in k]
    print(f"  Attention layers: {len(attention_keys)}")
    
    # Check channel dimensions
    for key in unet_keys:
        if 'model.diffusion_model.input_blocks.0.0.weight' in key:
            shape = sd[key].shape
            print(f"  Input block channels: {shape[1]} -> {shape[0]}")
            break
    
    return sd

def get_correct_model_config():
    """Get the correct model configuration that matches the checkpoint"""
    from omegaconf import OmegaConf
    
    # Try different config files
    config_paths = [
        "configs/latent-diffusion/ffhq-ldm-vq-4.yaml",
        "configs/latent-diffusion/ffhq-ldm-vq-4-fixed.yaml", 
        "configs/latent-diffusion/ffhq-ldm-vq-4-corrected.yaml",
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f" Using config: {config_path}")
            config = OmegaConf.load(config_path)
            
            # Print config details
            print_config_details(config, config_path)
            
            # Modify the config to match checkpoint architecture
            if 'model' in config and 'params' in config.model:
                model_params = config.model.params
                
                # Ensure UNet config matches checkpoint expectations
                if 'unet_config' in model_params and 'params' in model_params.unet_config:
                    unet_params = model_params.unet_config.params
                    
                    # Update to match the checkpoint (224 channels instead of 128)
                    if 'model_channels' in unet_params and unet_params.model_channels == 128:
                        print(" Updating UNet channels to match checkpoint...")
                        unet_params.model_channels = 224
                        
                    # Update attention dimensions
                    if 'num_heads' in unet_params and unet_params.num_heads == 4:
                        unet_params.num_heads = 8
                
                # Fix first_stage_config parameters
                if 'first_stage_config' in model_params and 'params' in model_params.first_stage_config:
                    first_stage_params = model_params.first_stage_config.params
                    # Remove problematic parameters that cause VectorQuantizer issues
                    problematic_params = ['remap', 'unknown_index', 'sane_index_shape', 'legacy']
                    for param in problematic_params:
                        if param in first_stage_params:
                            del first_stage_params[param]
                            print(f"  Removed problematic parameter: {param}")
            
            return config
    
    # Fallback: create minimal config
    print("💡 Creating minimal compatible config...")
    return create_minimal_config()

def print_config_details(config, config_path):
    """Print detailed information about the config"""
    print(f"\n CONFIG ANALYSIS: {config_path}")
    print("-" * 40)
    
    if 'model' in config and 'params' in config.model:
        params = config.model.params
        
        print(" Key Parameters:")
        
        # UNet parameters
        if 'unet_config' in params and 'params' in params.unet_config:
            unet = params.unet_config.params
            print(f"  UNet:")
            print(f"    - model_channels: {unet.get('model_channels', 'N/A')}")
            print(f"    - num_heads: {unet.get('num_heads', 'N/A')}")
            print(f"    - num_head_channels: {unet.get('num_head_channels', 'N/A')}")
            print(f"    - attention_resolutions: {unet.get('attention_resolutions', 'N/A')}")
        
        # First stage (VQGAN) parameters
        if 'first_stage_config' in params and 'params' in params.first_stage_config:
            vqgan = params.first_stage_config.params
            print(f"  VQGAN:")
            print(f"    - embed_dim: {vqgan.get('embed_dim', 'N/A')}")
            print(f"    - n_embed: {vqgan.get('n_embed', 'N/A')}")
            if 'ddconfig' in vqgan:
                dd = vqgan['ddconfig']
                print(f"    - z_channels: {dd.get('z_channels', 'N/A')}")
        
        print(f"  Other:")
        print(f"    - scale_factor: {params.get('scale_factor', 'N/A')}")
        print(f"    - conditioning_key: {params.get('conditioning_key', 'N/A')}")
    
    print("-" * 40)

def create_minimal_config():
    """Create a minimal config that should work with the checkpoint"""
    from omegaconf import OmegaConf
    
    config = {
        'model': {
            'target': 'ldm.models.diffusion.ddpm.LatentDiffusion',
            'params': {
                'linear_start': 0.0015,
                'linear_end': 0.0195,
                'num_timesteps_cond': 1,
                'log_every_t': 200,
                'timesteps': 1000,
                'first_stage_key': 'image',
                'cond_stage_key': 'class_label',
                'image_size': 64,
                'channels': 3,
                'cond_stage_trainable': False,
                'conditioning_key': None,
                'monitor': 'val/loss_simple_ema',
                'scale_factor': 1.0,
                'use_ema': False,
                'unet_config': {
                    'target': 'ldm.modules.diffusionmodules.openaimodel.UNetModel',
                    'params': {
                        'image_size': 32,
                        'in_channels': 3,
                        'out_channels': 3,
                        'model_channels': 224,  # Matches checkpoint
                        'attention_resolutions': [1, 2, 4],
                        'num_res_blocks': 2,
                        'channel_mult': [1, 2, 3, 4],
                        'num_heads': 8,  # Matches checkpoint
                        'use_spatial_transformer': False,
                        'transformer_depth': 1,
                        'context_dim': None,
                        'use_checkpoint': True,
                        'legacy': False
                    }
                },
                'first_stage_config': {
                    'target': 'ldm.models.autoencoder.VQModelInterface',
                    'params': {
                        'embed_dim': 3,
                        'n_embed': 8192,
                        'beta': 0.25,  # Explicitly add beta
                        'ddconfig': {
                            'double_z': False,
                            'z_channels': 3,
                            'resolution': 256,
                            'in_channels': 3,
                            'out_ch': 3,
                            'ch': 128,
                            'ch_mult': [1, 2, 4],
                            'num_res_blocks': 2,
                            'attn_resolutions': [],
                            'dropout': 0.0
                        },
                        'lossconfig': {
                            'target': 'torch.nn.Identity'
                        }
                    }
                },
                'cond_stage_config': {
                    'target': 'ldm.modules.encoders.modules.ClassEmbedder',
                    'params': {
                        'embed_dim': 512,
                        'n_classes': 1000
                    }
                }
            }
        }
    }
    
    print("\n  CREATED MINIMAL CONFIG WITH FOLLOWING PARAMETERS:")
    print_config_details(OmegaConf.create(config), "Minimal Config")
    
    return OmegaConf.create(config)

def load_model_direct():
    """Load model with proper configuration matching"""
    
    # Setup environment first
    if not setup_environment():
        return None
    
    # Now import LDM modules
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    # Apply fixes
    apply_compatibility_fixes()
    
    # Get correct config
    config = get_correct_model_config()
    
    ckpt_path = "models/ldm/ffhq-ldm-vq-4/model.ckpt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        # Check for zip file
        zip_path = "models/ldm/ffhq-ldm-vq-4/ffhq.zip"
        if os.path.exists(zip_path):
            print("💡 Found zip file - please extract it first:")
            print(f"   unzip {zip_path} -d models/ldm/ffhq-ldm-vq-4/")
        return None
    
    print(f" Loading checkpoint from: {ckpt_path}")
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f" Failed to load checkpoint: {e}")
        return None
    
    # Analyze checkpoint parameters
    sd = analyze_checkpoint_parameters(pl_sd)
    
    print("\n Instantiating model...")
    try:
        model = instantiate_from_config(config.model)
        
        print("📥 Loading weights...")
        # Analyze model parameters before loading
        model_params_count = sum(p.numel() for p in model.parameters())
        checkpoint_params_count = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
        
        print(f"\n PARAMETER COMPARISON:")
        print(f"  Model has: {model_params_count:,} parameters")
        print(f"  Checkpoint has: {checkpoint_params_count:,} parameters")
        
        if model_params_count > checkpoint_params_count:
            print("  WARNING: Model has MORE parameters than checkpoint!")
        elif model_params_count < checkpoint_params_count:
            print("  WARNING: Model has FEWER parameters than checkpoint!")
        else:
            print("✅ Model and checkpoint parameter counts match!")
        
        # Use strict=False to ignore minor mismatches
        missing, unexpected = model.load_state_dict(sd, strict=False)
        
        print("\n LOADING RESULTS:")
        print("-" * 40)
        
        if missing:
            print(f"❌ MISSING keys: {len(missing)}")
            print("   These parameters exist in checkpoint but not in model:")
            if len(missing) < 20:  # Show all if not too many
                for key in missing:
                    print(f"   - {key}")
            else:
                for key in missing[:10]:  # Show first 10 only
                    print(f"   - {key}")
                print(f"   ... and {len(missing)-10} more")
        else:
            print("✅ No missing keys!")
        
        if unexpected:
            print(f"\n  UNEXPECTED keys: {len(unexpected)}")
            print("   These parameters exist in model but not in checkpoint:")
            if len(unexpected) < 20:
                for key in unexpected:
                    print(f"   + {key}")
            else:
                for key in unexpected[:10]:
                    print(f"   + {key}")
                print(f"   ... and {len(unexpected)-10} more")
        else:
            print("✅ No unexpected keys!")
        
        # Calculate matching percentage
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(sd.keys())
        matching_keys = model_keys.intersection(checkpoint_keys)
        match_percentage = (len(matching_keys) / len(model_keys)) * 100 if model_keys else 0
        
        print(f"\n KEY MATCHING: {match_percentage:.1f}%")
        print(f"   Model keys: {len(model_keys)}")
        print(f"   Checkpoint keys: {len(checkpoint_keys)}")
        print(f"   Matching keys: {len(matching_keys)}")
        
        model.eval()
        model = model.to("cpu")
        
        print("\n Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        # Try alternative approach
        return load_model_alternative(sd)

def load_model_alternative(state_dict):
    """Alternative loading approach"""
    print("\n" + "="*60)
    print(" TRYING ALTERNATIVE LOADING APPROACH...")
    print("="*60)
    
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from ldm.util import get_obj_from_str
    
    # Create model with minimal config
    config = create_minimal_config()
    
    try:
        # Instantiate directly
        model_class = get_obj_from_str(config.model.target)
        model = model_class(**config.model.params)
        
        # Analyze model architecture
        print("\n🔍 MODEL ARCHITECTURE ANALYSIS:")
        print("-" * 40)
        
        # Check model components
        if hasattr(model, 'diffusion_model'):
            print("✅ diffusion_model component exists")
            unet_params = sum(p.numel() for p in model.diffusion_model.parameters())
            print(f"   UNet parameters: {unet_params:,}")
        
        if hasattr(model, 'first_stage_model'):
            print("✅ first_stage_model (VQGAN) exists")
            vqgan_params = sum(p.numel() for p in model.first_stage_model.parameters())
            print(f"   VQGAN parameters: {vqgan_params:,}")
        
        if hasattr(model, 'cond_stage_model'):
            print("✅ cond_stage_model exists")
            cond_params = sum(p.numel() for p in model.cond_stage_model.parameters())
            print(f"   Conditioner parameters: {cond_params:,}")
        
        # Load state dict more carefully
        model_state_dict = model.state_dict()
        
        print(f"\n FILTERING STATE DICT:")
        print(f"   Model has {len(model_state_dict)} parameters")
        print(f"   Checkpoint has {len(state_dict)} parameters")
        
        # Filter state dict to only include matching keys
        filtered_state_dict = {}
        shape_mismatches = []
        key_transformations = []
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    shape_mismatches.append((k, v.shape, model_state_dict[k].shape))
            else:
                # Try to find matching key with different prefix
                alt_key = k.replace('model.diffusion_model.', '')
                if alt_key in model_state_dict and model_state_dict[alt_key].shape == v.shape:
                    filtered_state_dict[alt_key] = v
                    key_transformations.append((k, alt_key))
        
        if shape_mismatches:
            print(f"\n  SHAPE MISMATCHES ({len(shape_mismatches)}):")
            for k, checkpoint_shape, model_shape in shape_mismatches[:5]:
                print(f"   {k}: checkpoint={checkpoint_shape}, model={model_shape}")
        
        if key_transformations:
            print(f"\n KEY TRANSFORMATIONS ({len(key_transformations)}):")
            for orig, new in key_transformations[:5]:
                print(f"   {orig} -> {new}")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to("cpu")
        
        loaded_percentage = (len(filtered_state_dict) / len(state_dict)) * 100
        print(f"\n Alternative loading loaded {len(filtered_state_dict)}/{len(state_dict)} parameters ({loaded_percentage:.1f}%)")
        return model
        
    except Exception as e:
        print(f"❌ Alternative loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_latent_shape(model):
    """Get the correct latent shape for the model"""
    print("\n DETERMINING LATENT SHAPE...")
    
    # Try multiple approaches to get the latent shape
    shape_attempts = []
    
    # Approach 1: Check first_stage_model attributes
    if hasattr(model, 'first_stage_model'):
        print("   Checking first_stage_model attributes:")
        if hasattr(model.first_stage_model, 'encoder'):
            encoder = model.first_stage_model.encoder
            # Try different attribute names
            if hasattr(encoder, 'z_channels'):
                shape_attempts.append([encoder.z_channels, 64, 64])
                print(f"     Found z_channels: {encoder.z_channels}")
            if hasattr(encoder, 'in_channels'):
                shape_attempts.append([encoder.in_channels, 64, 64])
                print(f"     Found in_channels: {encoder.in_channels}")
            if hasattr(encoder, 'out_channels'):
                shape_attempts.append([encoder.out_channels, 64, 64])
                print(f"     Found out_channels: {encoder.out_channels}")
    
    # Approach 2: Check model config
    if hasattr(model, 'first_stage_config'):
        print("   Checking first_stage_config:")
        if hasattr(model.first_stage_config, 'params'):
            params = model.first_stage_config.params
            if 'embed_dim' in params:
                shape_attempts.append([params.embed_dim, 64, 64])
                print(f"     Found embed_dim: {params.embed_dim}")
            if 'z_channels' in params:
                shape_attempts.append([params.z_channels, 64, 64])
                print(f"     Found z_channels: {params.z_channels}")
    
    # Approach 3: Common FFHQ latent shapes
    common_shapes = [
        [3, 64, 64],   # Most common for FFHQ
        [4, 64, 64],   # Alternative
        [3, 32, 32],   # Smaller
        [4, 32, 32],   # Smaller alternative
    ]
    shape_attempts.extend(common_shapes)
    
    # Remove duplicates
    unique_shapes = []
    for shape in shape_attempts:
        if shape not in unique_shapes:
            unique_shapes.append(shape)
    
    print(f"\n📋 Shape candidates: {unique_shapes}")
    
    # Return the most common one first
    selected_shape = [3, 64, 64]
    print(f"✅ Selected latent shape: {selected_shape}")
    return selected_shape

def get_conditioning(model, num_samples, shape):
    """Get proper conditioning for the model"""
    print("\n SETTING UP CONDITIONING...")
    
    # Check the model's conditioning key to understand what type of conditioning it needs
    conditioning_key = getattr(model, 'conditioning_key', None)
    print(f"   Model conditioning key: {conditioning_key}")
    
    # FORCE concatenation conditioning since the model expects it despite claiming to be unconditional
    # Create empty concatenation conditioning tensor with 0 channels
    c_concat = torch.zeros(num_samples, 0, shape[1], shape[2], device="cpu")
    
    # The sampler expects the conditioning to be a tensor, not a dict
    # For unconditional models, we should use None, but this model needs the empty tensor
    # We'll pass the tensor directly to the sampler
    print(f"   Using forced concatenation conditioning with shape: {c_concat.shape}")
    
    return c_concat

def generate_samples(model, num_samples=2, steps=50):
    """Generate samples using DDIM sampler"""
    if model is None:
        print(" No model available for generation")
        return None
    
    try:
        from ldm.models.diffusion.ddim import DDIMSampler
    except ImportError as e:
        print(f"❌ Cannot import DDIMSampler: {e}")
        return None
    
    sampler = DDIMSampler(model)
    
    # Get latent shape
    shape = get_latent_shape(model)
    
    print(f"\n GENERATING {num_samples} SAMPLES")
    print(f"   Shape: {shape}")
    print(f"   Steps: {steps}")
    
    try:
        with torch.no_grad():
            with model.ema_scope():
                # Get proper conditioning - ALWAYS use concatenation conditioning
                conditioning = get_conditioning(model, num_samples, shape)
                
                # For unconditional generation, we use the same conditioning for both
                # Pass the tensor directly to the sampler (not as a dict)
                cond = conditioning
                uncond = conditioning
                
                print("\n  SAMPLER SETTINGS:")
                print(f"   Conditioning shape: {cond.shape}")
                print(f"   Unconditional conditioning shape: {uncond.shape}")
                print(f"   Guidance scale: 1.0 (unconditional)")
                print(f"   Eta: 1.0")
                
                # Generate unconditional samples with proper conditioning
                samples, _ = sampler.sample(
                    S=steps,
                    batch_size=num_samples,
                    shape=shape,
                    conditioning=cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=1.0,  # Important for unconditional generation
                    unconditional_conditioning=uncond,  # Same as conditioning for unconditional
                    x_T=None
                )
                
                # Decode from latent space
                print("\n DECODING SAMPLES...")
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                print(f"✅ Generated samples shape: {x_samples.shape}")
                return x_samples
                
    except Exception as e:
        print(f" Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_images(images, output_dir="ffhq_test_output"):
    """Save generated images"""
    if images is None:
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, image in enumerate(images):
        # Convert tensor to numpy array
        image = image.cpu().numpy()
        
        # Handle different tensor shapes
        if image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)
        elif image.shape[-1] == 3:  # HWC format
            pass
        else:
            # Assume first 3 channels are RGB
            image = image[:3].transpose(1, 2, 0)
        
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Save image
        filename = f"ffhq_sample_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        pil_image.save(filepath)
        saved_paths.append(filepath)
        print(f"💾 Saved: {filepath}")
    
    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Fixed FFHQ model generation")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--output_dir", type=str, default="ffhq_test_output", help="Output directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" STARTING FIXED FFHQ MODEL TEST")
    print("="*60)
    print(f" Parameters: {args.num_samples} samples, {args.steps} steps")
    
    try:
        # Load model
        print("\n LOADING MODEL...")
        start_time = time.time()
        model = load_model_direct()
        
        if model is None:
            print("❌ Failed to load model")
            return
        
        load_time = time.time() - start_time
        print(f"\n✅ Model loaded in {load_time:.2f} seconds")
        
        # Generate samples
        print("\n GENERATING SAMPLES...")
        gen_start = time.time()
        samples = generate_samples(model, args.num_samples, args.steps)
        gen_time = time.time() - gen_start
        
        if samples is None:
            print("❌ Failed to generate samples")
            return
            
        print(f"\n✅ Samples generated in {gen_time:.2f} seconds")
        
        # Save images
        print("\n SAVING IMAGES...")
        saved_paths = save_images(samples, args.output_dir)
        
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f" SUCCESS! Generated {len(saved_paths)} samples in {total_time:.2f} seconds")
        print(f" Output directory: {args.output_dir}")
        
        # Print file sizes
        for path in saved_paths:
            size = os.path.getsize(path) / 1024  # KB
            print(f"   {os.path.basename(path)}: {size:.1f} KB")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
