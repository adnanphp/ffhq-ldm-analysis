# ffhq_diverse_generation_fixed.py
import torch
import os
import sys
import numpy as np
from PIL import Image
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Add latent-diffusion directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class DiverseFaceGenerator:
    def __init__(self):
        print("🎭 DIVERSE FFHQ FACE GENERATOR")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"diverse_faces_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.faces_dir = os.path.join(self.output_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        # Load the model using YOUR WORKING CODE
        self.model = None
        self.sampler = None
        self.load_model()
    
    def load_model(self):
        """Load the model using YOUR WORKING CONFIG from ffhq_lod_image_gen.py"""
        print("🔄 Loading model...")
        
        try:
            # Use YOUR working config
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
            config_path = os.path.join(self.output_dir, "ffhq_config.yaml")
            with open(config_path, "w") as f:
                f.write(config_yaml)
            
            # Load config
            config = OmegaConf.load(config_path)
            
            # Create model
            self.model = instantiate_from_config(config.model)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Load checkpoint (UPDATE THIS PATH!)
            ckpt_path = "models/ldm/ffhq-ldm-vq-4/model.ckpt"
            
            # If checkpoint doesn't exist, try alternatives
            if not os.path.exists(ckpt_path):
                print(f"⚠️  Checkpoint not found at: {ckpt_path}")
                print("   Looking for alternative checkpoints...")
                
                # Try to find any checkpoint
                possible_paths = [
                    "models/ldm/ffhq256/model.ckpt",
                    "latent-diffusion/models/ldm/ffhq256/model.ckpt",
                    "model.ckpt",
                    "v2-1_512-ema-pruned.ckpt",  # SD 2.1
                    "sd-v1-5.ckpt",  # SD 1.5
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        ckpt_path = path
                        print(f"✅ Found checkpoint: {path}")
                        break
            
            print(f"📂 Loading checkpoint: {ckpt_path}")
            
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ Checkpoint loaded")
            else:
                print("⚠️  No checkpoint found. Using random weights.")
                print("⚠️  Faces may not look realistic.")
            
            # Create sampler
            self.sampler = DDIMSampler(self.model)
            
            print("✅ Model loaded successfully")
            
            # Remove temp config
            os.remove(config_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Make sure you have the model checkpoint")
            print("2. Update ckpt_path in the code to your actual checkpoint location")
            print("3. Or run without model (analyze existing faces only)")
            return False
    
    def generate_single_face(self, seed=42, guidance_scale=7.5, steps=10): # (self, seed=42, guidance_scale=7.5, steps=10):  # CHANGE 150 → 10
        """Generate a single face with specific parameters"""
        if self.model is None or self.sampler is None:
            print("❌ Model not loaded")
            return None
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Settings from YOUR working code
        shape = [3, 64, 64]  # Latent shape
        c = torch.zeros(1, 0, 64, 64).to(self.device)  # Unconditional
        uc = torch.zeros(1, 0, 64, 64).to(self.device)
        
        # Generate
        with torch.no_grad():
            samples, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                eta=0.0,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
            )
            
            # Decode
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
        
        # Convert to numpy
        img_np = x_samples[0].cpu().numpy().transpose(1, 2, 0)
        
        return img_np
    
    def generate_diverse_batch(self, n_faces=5, base_seed=42, guidance_scale=7.5):
        """Generate a batch of diverse faces with different seeds"""
        print(f"\n GENERATING {n_faces} DIVERSE FACES")
        print("=" * 5)
        print(f"Base seed: {base_seed}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Each face gets a unique seed")
        
        faces = []
        
        for i in range(n_faces):
            # Use DIFFERENT seed for each face (CRITICAL FOR DIVERSITY!)
            seed = base_seed + (i * 100)  # Different seeds spaced apart
            
            print(f"  Generating face {i+1}/{n_faces} with seed {seed}...", end='\r')
            
            try:
                # Generate face
                face_np = self.generate_single_face(
                    seed=seed,
                    guidance_scale=guidance_scale,
                    steps=150
                )
                
                if face_np is not None:
                    # Save image
                    face_img = Image.fromarray((face_np * 255).astype(np.uint8))
                    img_path = os.path.join(self.faces_dir, f"face_{i:03d}_seed{seed}.png")
                    face_img.save(img_path)
                    
                    # Store data
                    face_data = {
                        'id': i,
                        'seed': seed,
                        'guidance_scale': guidance_scale,
                        'filename': f"face_{i:03d}_seed{seed}.png",
                        'filepath': img_path,
                        'image_np': face_np,
                        'image': face_img
                    }
                    
                    faces.append(face_data)
                    
            except Exception as e:
                print(f"❌ Error generating face {i}: {e}")
        
        print(f"\n✅ Generated {len(faces)} faces")
        
        # Save metadata
        metadata = {
            'total_faces': len(faces),
            'base_seed': base_seed,
            'guidance_scale': guidance_scale,
            'generation_date': datetime.now().isoformat(),
            'seeds_used': [f['seed'] for f in faces],
            'output_dir': self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, "generation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata saved: {metadata_path}")
        
        return faces
    
    def analyze_novelty(self, faces):
        """Analyze novelty of generated faces"""
        print(f"\n📊 ANALYZING NOVELTY")
        print("=" * 40)
        
        if len(faces) < 2:
            print("❌ Need at least 2 faces for novelty analysis")
            return None
        
        # Extract features
        features = []
        for face in faces:
            img_np = face['image_np']
            
            # Simple features
            color_features = img_np.mean(axis=(0, 1)).flatten()
            brightness = img_np.mean()
            contrast = img_np.std()
            
            # Edge features
            gray = np.mean(img_np, axis=2)
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.hypot(sobel_x, sobel_y)
            edge_mean = edge_magnitude.mean()
            
            feature_vector = np.concatenate([
                color_features,
                np.array([brightness, contrast, edge_mean])
            ])
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(features)
        np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1.0
        
        # Calculate novelty metrics
        n_faces = len(faces)
        mask = ~np.eye(n_faces, dtype=bool)
        
        avg_similarity = similarity_matrix[mask].mean()
        max_similarity = similarity_matrix[mask].max()
        
        # Count unique faces (similarity < 0.9)
        n_unique = 0
        for i in range(n_faces):
            is_unique = True
            for j in range(n_faces):
                if i != j and similarity_matrix[i, j] > 0.9:  # 90% similarity threshold
                    is_unique = False
                    break
            if is_unique:
                n_unique += 1
        
        novelty_percentage = (n_unique / n_faces) * 100
        
        print(f"📈 NOVELTY RESULTS:")
        print(f"  Total faces analyzed: {n_faces}")
        print(f"  Unique faces: {n_unique}")
        print(f"  Novelty percentage: {novelty_percentage:.1f}%")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Maximum similarity: {max_similarity:.3f}")
        
        # Create visualization
        self.create_novelty_visualization(faces, similarity_matrix, novelty_percentage, avg_similarity)
        
        # Save results
        results = {
            'novelty_percentage': float(novelty_percentage),
            'avg_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'n_faces': n_faces,
            'n_unique': n_unique,
            'seeds_used': [f['seed'] for f in faces]
        }
        
        results_path = os.path.join(self.output_dir, "novelty_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved: {results_path}")
        
        return results
    
    def create_novelty_visualization(self, faces, similarity_matrix, novelty_percentage, avg_similarity):
        """Create visualization of novelty results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Similarity matrix heatmap
        im = ax1.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Face Similarity Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Face Index', fontsize=10)
        ax1.set_ylabel('Face Index', fontsize=10)
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Novelty gauge
        ax2 = plt.subplot(132, projection='polar')
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Score arc
        score = novelty_percentage / 100
        score_angle = score * np.pi
        score_theta = np.linspace(0, score_angle, 100)
        score_r = np.ones_like(score_theta)
        
        # Color based on score
        if score > 0.7:
            gauge_color = 'green'
        elif score > 0.5:
            gauge_color = 'orange'
        else:
            gauge_color = 'red'
        
        ax2.plot(score_theta, score_r, color=gauge_color, linewidth=8)
        ax2.plot(theta, r, color='gray', linewidth=8, alpha=0.3)
        
        # Add needle
        ax2.plot([score_angle, score_angle], [0.7, 1.1], color='black', linewidth=2)
        
        # Labels
        ax2.text(np.pi/2, 0.5, f'{novelty_percentage:.1f}%', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        if score > 0.7:
            rating = "EXCELLENT"
        elif score > 0.5:
            rating = "GOOD"
        elif score > 0.3:
            rating = "MODERATE"
        else:
            rating = "LOW"
        
        ax2.text(np.pi/2, 0.3, rating, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax2.set_xlim([0, np.pi])
        ax2.set_ylim([0, 1.2])
        ax2.axis('off')
        ax2.set_title('Novelty Score', fontsize=12, fontweight='bold')
        
        # 3. Similarity distribution
        n_faces = len(faces)
        mask = ~np.eye(n_faces, dtype=bool)
        similarities = similarity_matrix[mask]
        
        ax3.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=avg_similarity, color='red', linestyle='--', 
                   label=f'Avg: {avg_similarity:.3f}')
        ax3.axvline(x=0.9, color='orange', linestyle=':', 
                   label='Threshold: 0.9')
        
        ax3.set_xlabel('Similarity Score', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('Similarity Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Novelty Analysis: {novelty_percentage:.1f}% Unique Faces', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        viz_path = os.path.join(self.output_dir, "novelty_analysis.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved: {viz_path}")
        
        return viz_path
    
    def check_targets(self, novelty_results):
        """Check if we meet Week 1 targets"""
        print(f"\n🎯 WEEK 1 TARGET CHECK")
        print("=" * 40)
        
        if novelty_results is None:
            print("❌ No novelty results to check")
            return False
        
        novelty_pct = novelty_results['novelty_percentage']
        avg_sim = novelty_results['avg_similarity']
        
        print(f"Target 1: Novelty > 50%")
        print(f"  Your score: {novelty_pct:.1f}% → {'✅ PASS' if novelty_pct > 50 else '❌ FAIL'}")
        
        print(f"\nTarget 2: Average similarity < 0.8")
        print(f"  Your score: {avg_sim:.3f} → {'✅ PASS' if avg_sim < 0.8 else '❌ FAIL'}")
        
        print(f"\nTarget 3: Both targets met")
        both_met = novelty_pct > 50 and avg_sim < 0.8
        print(f"  Status: {'✅ ALL TARGETS MET!' if both_met else '❌ TARGETS NOT MET'}")
        
        # Save target check
        target_check = {
            'week1_targets': {
                'novelty_gt_50': novelty_pct > 50,
                'avg_similarity_lt_08': avg_sim < 0.8,
                'both_met': both_met
            },
            'your_scores': {
                'novelty_percentage': novelty_pct,
                'avg_similarity': avg_sim
            },
            'check_date': datetime.now().isoformat()
        }
        
        target_path = os.path.join(self.output_dir, "target_check.json")
        with open(target_path, 'w') as f:
            json.dump(target_check, f, indent=2)
        
        print(f"\n📄 Target check saved: {target_path}")
        
        return both_met
    
    def run_complete_experiment(self, n_faces=5, guidance_scale=7.5):
        """Run complete experiment: generate and analyze"""
        print("\n🔬 RUNNING COMPLETE EXPERIMENT")
        print("=" * 50)
        
        # Step 1: Generate faces
        faces = self.generate_diverse_batch(
            n_faces=n_faces,
            guidance_scale=guidance_scale,
            base_seed=42
        )
        
        if len(faces) < 10:
            print("❌ Not enough faces generated. Experiment failed.")
            return False
        
        # Step 2: Analyze novelty
        novelty_results = self.analyze_novelty(faces)
        
        # Step 3: Check targets
        targets_met = self.check_targets(novelty_results)
        
        # Step 4: Create summary
        self.create_experiment_summary(faces, novelty_results, targets_met, guidance_scale)
        
        print(f"\n✅ EXPERIMENT COMPLETE!")
        print(f"📁 All results saved in: {self.output_dir}")
        
        if targets_met:
            print(f"\n🎉 CONGRATULATIONS! You've fixed the 0% novelty problem!")
            print(f"   Novelty: {novelty_results['novelty_percentage']:.1f}% (was 0%)")
            print(f"   Average similarity: {novelty_results['avg_similarity']:.3f} (was 98.4%)")
        else:
            print(f"\n⚠️  Targets not met. Try:")
            print(f"   1. Increase guidance scale (try 10.0 or 15.0)")
            print(f"   2. Generate more faces (try 100+)")
            print(f"   3. Use seeds further apart (seed * 1000)")
        
        return targets_met
    
    def create_experiment_summary(self, faces, novelty_results, targets_met, guidance_scale):
        """Create experiment summary"""
        summary = f"""
        DIVERSE FACE GENERATION EXPERIMENT
        ==================================
        
        Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Output Directory: {self.output_dir}
        
        GENERATION PARAMETERS:
        - Number of faces: {len(faces)}
        - Guidance scale: {guidance_scale}
        - Base seed: 42 (each face gets seed + i*100)
        - Steps per face: 150
        - Device: {self.device}
        
        NOVELTY RESULTS:
        - Novelty percentage: {novelty_results['novelty_percentage']:.1f}%
        - Average similarity: {novelty_results['avg_similarity']:.3f}
        - Maximum similarity: {novelty_results['max_similarity']:.3f}
        - Unique faces: {novelty_results['n_unique']} of {novelty_results['n_faces']}
        
        WEEK 1 TARGETS:
        - Target: Novelty > 50%
        - Result: {novelty_results['novelty_percentage']:.1f}% → {'PASS' if novelty_results['novelty_percentage'] > 50 else 'FAIL'}
        
        - Target: Average similarity < 0.8
        - Result: {novelty_results['avg_similarity']:.3f} → {'PASS' if novelty_results['avg_similarity'] < 0.8 else 'FAIL'}
        
        OVERALL: {'ALL TARGETS MET! 🎉' if targets_met else 'TARGETS NOT MET ⚠️'}
        
        FILES GENERATED:
        - Faces: {self.faces_dir}/ (PNG files)
        - Metadata: {self.output_dir}/generation_metadata.json
        - Novelty results: {self.output_dir}/novelty_results.json
        - Target check: {self.output_dir}/target_check.json
        - Visualization: {self.output_dir}/novelty_analysis.png
        
        NEXT STEPS:
        1. Check the generated faces in {self.faces_dir}/
        2. If novelty is still low, try guidance_scale=10.0 or 15.0
        3. Run the full advanced analysis: python ffhq_advanced_diversity_analysis_fixed_v2.py
        """
        
        summary_path = os.path.join(self.output_dir, "experiment_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"📋 Summary saved: {summary_path}")
        
        return summary_path

def main():
    """Main function"""
    print("🎭 DIVERSE FFHQ FACE GENERATION & NOVELTY ANALYSIS")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Generate diverse faces using DIFFERENT random seeds")
    print("2. Calculate novelty percentage (was 0%)")
    print("3. Check Week 1 targets: Novelty > 50%, Similarity < 0.8")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_faces', type=int, default=5,
                       help='Number of faces to generate (default: 5)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale (default: 7.5, try 10.0 for more diversity)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with 20 faces')
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick_test:
        args.n_faces = 5
        print("🚀 Quick test mode: generating 5 faces")
    
    print(f"\n📊 Experiment setup:")
    print(f"  Number of faces: {args.n_faces}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Each face will have a UNIQUE random seed")
    print(f"\n🚀 Starting experiment...")
    
    # Initialize and run
    generator = DiverseFaceGenerator()
    
    if generator.model is None:
        print("\n⚠️  Model failed to load. Trying to continue with what we have...")
        print("   You may need to manually update the checkpoint path in the code.")
        print("   Looking for: models/ldm/ffhq-ldm-vq-4/model.ckpt")
        print("\n   Continuing with existing faces if any...")
    
    # Run experiment
    success = generator.run_complete_experiment(
        n_faces=args.n_faces,
        guidance_scale=args.guidance_scale
    )
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    if success:
        print("🎉 SUCCESS! You should now have NON-ZERO novelty!")
    else:
        print("⚠️  Some issues encountered. Check the output directory.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
