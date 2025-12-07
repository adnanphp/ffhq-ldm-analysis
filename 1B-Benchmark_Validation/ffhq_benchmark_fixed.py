# ffhq_benchmark_fixed.py
import torch
import os
import sys
import numpy as np
from PIL import Image
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# Add the latent-diffusion directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def create_simple_config():
    """Create simple config for unconditional generation"""
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
    conditioning_key: null
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
    
    config_path = "ffhq_simple_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)
    
    print(f"✅ Created config file: {config_path}")
    return config_path

class FFHQBenchmarkFixed:
    def __init__(self, model_path="models/ldm/ffhq-ldm-vq-4/model.ckpt"):
        """Initialize the benchmark class"""
        print("🎭 FFHQ PERFORMANCE BENCHMARK (FIXED)")
        print("=" * 60)
        
        # Create config
        self.config_path = create_simple_config()
        self.model_path = model_path
        
        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        if self.device.type == "cpu":
            print("  WARNING: Running on CPU - this will be VERY SLOW!")
            print("If you have a GPU, make sure PyTorch CUDA is installed")
        
        # Load model
        print(" Loading model...")
        self.config = OmegaConf.load(self.config_path)
        self.model = instantiate_from_config(self.config.model)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Create sampler with progress disabled for benchmarking
        self.sampler = DDIMSampler(self.model)
        
        # Create output directory
        self.output_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f" Model loaded")
        print(f" Output directory: {self.output_dir}")
        
        # Performance results storage
        self.results = []
    
    def clear_memory(self):
        """Clear GPU and CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def save_images(self, images, batch_size, iteration, step_idx=None):
        """Save generated images to disk"""
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        
        # Create batch directory
        batch_dir = os.path.join(self.output_dir, f"batch_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save each image
        saved_files = []
        for i in range(images_np.shape[0]):
            # Generate filename
            if step_idx is not None:
                filename = f"batch{batch_size}_iter{iteration}_step{step_idx}_img{i}.png"
            else:
                filename = f"batch{batch_size}_iter{iteration}_img{i}.png"
            
            filepath = os.path.join(batch_dir, filename)
            
            # Convert and save
            img_array = (images_np[i] * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(filepath)
            
            saved_files.append(filepath)
        
        return saved_files
    
    def measure_inference_speed(self, batch_sizes=[1, 2, 4], steps=20, num_repeats=2):
        """
        Measure inference speed for different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            steps: Number of diffusion steps (reduced for faster testing)
            num_repeats: Number of repetitions for each batch size
        """
        print("\n" + "="*60)
        print("INFERENCE SPEED BENCHMARK")
        print("=" * 60)
        
        print(f"\n Settings:")
        print(f"  Steps: {steps} (reduced for faster testing)")
        print(f"  Repeats per batch: {num_repeats}")
        print(f"  Device: {self.device}")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Output directory: {self.output_dir}")
        
        # Create results CSV file
        results_file = os.path.join(self.output_dir, "benchmark_results.csv")
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_size', 'iteration', 'total_time', 'sampling_time', 
                           'decoding_time', 'images_per_sec', 'images_saved'])
        
        all_results = []
        
        for batch_size in batch_sizes:
            print(f"\n{'='*40}")
            print(f"Testing batch size: {batch_size}")
            print(f"{'='*40}")
            
            batch_times = []
            
            for repeat in range(num_repeats):
                print(f"\n  Repeat {repeat+1}/{num_repeats}...")
                
                # Clear memory between runs
                self.clear_memory()
                
                # Generate with timing
                start_time = time.time()
                
                # Prepare tensors
                shape = [3, 64, 64]
                c = torch.zeros(batch_size, 0, 64, 64).to(self.device)
                uc = torch.zeros(batch_size, 0, 64, 64).to(self.device)
                
                # Generate samples
                samples, _ = self.sampler.sample(
                    S=steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    eta=0.0,
                    verbose=False,
                    unconditional_guidance_scale=7.5,
                    unconditional_conditioning=uc,
                )
                
                sampling_time = time.time() - start_time
                
                # Decode samples to images
                decode_start = time.time()
                with torch.no_grad():
                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
                decode_time = time.time() - decode_start
                
                total_time = time.time() - start_time
                
                # Save images
                saved_files = self.save_images(x_samples, batch_size, repeat)
                
                # Record times
                batch_times.append({
                    'total': total_time,
                    'sampling': sampling_time,
                    'decoding': decode_time,
                    'images_per_sec': batch_size / total_time,
                    'images_saved': len(saved_files)
                })
                
                print(f"    Generated and saved {len(saved_files)} images")
                print(f"      Total: {total_time:.2f}s | "
                      f"Sampling: {sampling_time:.2f}s | "
                      f"Decoding: {decode_time:.2f}s")
                print(f"     Speed: {batch_size/total_time:.2f} img/s")
                
                # Save to CSV
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        batch_size,
                        repeat,
                        f"{total_time:.4f}",
                        f"{sampling_time:.4f}",
                        f"{decode_time:.4f}",
                        f"{batch_size/total_time:.4f}",
                        len(saved_files)
                    ])
            
            # Calculate statistics for this batch size
            avg_total = np.mean([t['total'] for t in batch_times])
            avg_sampling = np.mean([t['sampling'] for t in batch_times])
            avg_decoding = np.mean([t['decoding'] for t in batch_times])
            avg_img_per_sec = np.mean([t['images_per_sec'] for t in batch_times])
            
            result = {
                'batch_size': batch_size,
                'avg_total_time': avg_total,
                'avg_sampling_time': avg_sampling,
                'avg_decoding_time': avg_decoding,
                'avg_images_per_sec': avg_img_per_sec,
                'total_images_generated': batch_size * num_repeats
            }
            
            all_results.append(result)
            
            print(f"\n   Summary for batch size {batch_size}:")
            print(f"  Average time: {avg_total:.2f}s")
            print(f"    Average speed: {avg_img_per_sec:.2f} img/s")
            print(f"    Total images generated: {batch_size * num_repeats}")
            print(f"    Images saved in: {os.path.join(self.output_dir, f'batch_{batch_size}')}")
        
        self.results = all_results
        print(f"\n✅ All results saved to: {results_file}")
        return all_results
    
    def create_sample_grid(self, batch_size=4, steps=50):
        """Create a beautiful grid of sample images"""
        print(f"\n{'='*60}")
        print(f"CREATING SAMPLE GRID")
        print(f"{'='*60}")
        
        self.clear_memory()
        
        # Generate images
        print(f"Generating {batch_size} sample faces...")
        start_time = time.time()
        
        shape = [3, 64, 64]
        c = torch.zeros(batch_size, 0, 64, 64).to(self.device)
        uc = torch.zeros(batch_size, 0, 64, 64).to(self.device)
        
        samples, _ = self.sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            eta=0.0,
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
        )
        
        with torch.no_grad():
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
        
        gen_time = time.time() - start_time
        
        # Convert to numpy
        images_np = x_samples.cpu().numpy().transpose(0, 2, 3, 1)
        
        print(f"✅ Generated {batch_size} faces in {gen_time:.2f}s")
        
        # Create grid layout
        cols = min(4, batch_size)
        rows = (batch_size + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each image
        for idx in range(batch_size):
            row = idx // cols
            col = idx % cols
            
            axes[row, col].imshow(images_np[idx])
            axes[row, col].set_title(f'Face {idx+1}', fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
            
            # Add quality metrics
            img = images_np[idx]
            contrast = img.std()
            brightness = img.mean()
            
            # Simple face detection heuristics
            r, g, b = img.mean(axis=(0, 1))
            is_face_like = (contrast > 0.2) and (r > g > b)
            
            quality = "✅ Face-like" if is_face_like else " Other"
            color = "green" if is_face_like else "orange"
            
            axes[row, col].text(0.02, 0.98, quality,
                              transform=axes[row, col].transAxes,
                              fontsize=9, color=color, fontweight='bold',
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(batch_size, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Add title and info
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        plt.suptitle(f'FFHQ Generated Faces\nBatch Size: {batch_size} | Device: {device_info} | Time: {gen_time:.1f}s', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the grid
        grid_filename = os.path.join(self.output_dir, "sample_grid.png")
        plt.savefig(grid_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save individual images
        individual_dir = os.path.join(self.output_dir, "individual_samples")
        os.makedirs(individual_dir, exist_ok=True)
        
        for idx in range(batch_size):
            img_filename = os.path.join(individual_dir, f"face_{idx+1}.png")
            img_array = (images_np[idx] * 255).astype(np.uint8)
            Image.fromarray(img_array).save(img_filename)
        
        print(f"\n Output files:")
        print(f"  • Sample grid: {grid_filename}")
        print(f"  • Individual faces: {individual_dir}/")
        print(f"  • Benchmark results: {os.path.join(self.output_dir, 'benchmark_results.csv')}")
        
        return images_np
    
    def plot_performance(self):
        """Create performance visualization"""
        if not self.results:
            print("❌ No results to plot")
            return
        
        batch_sizes = [r['batch_size'] for r in self.results]
        images_per_sec = [r['avg_images_per_sec'] for r in self.results]
        total_times = [r['avg_total_time'] for r in self.results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Throughput
        ax1.plot(batch_sizes, images_per_sec, 'b-o', linewidth=3, markersize=10)
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Images per Second', fontsize=12)
        ax1.set_title('Generation Throughput', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(batch_sizes)
        
        # Add value labels on points
        for x, y in zip(batch_sizes, images_per_sec):
            ax1.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Total time
        bars = ax2.bar([str(bs) for bs in batch_sizes], total_times, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Time per Batch (seconds)', fontsize=12)
        ax2.set_title('Generation Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        plt.suptitle(f'FFHQ Model Performance on {device}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(self.output_dir, "performance_plot.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance plot saved: {plot_filename}")
    
    def print_final_report(self):
        """Print final report"""
        print("\n" + "="*60)
        print("FINAL BENCHMARK REPORT")
        print("=" * 60)
        
        print(f"\n📊 Results Summary:")
        print(f"{'Batch Size':<12} {'Img/s':<12} {'Time/Batch':<12} {'Efficiency':<12}")
        print("-" * 60)
        
        for result in self.results:
            efficiency = (result['avg_images_per_sec'] / result['batch_size']) * 100
            print(f"{result['batch_size']:<12} "
                  f"{result['avg_images_per_sec']:<12.2f} "
                  f"{result['avg_total_time']:<12.2f} "
                  f"{efficiency:<12.1f}%")
        
        print(f"\n All files saved in: {self.output_dir}/")
        print(f"\n💡 Next steps:")
        print("  1. Check the 'individual_samples' folder for generated faces")
        print("  2. Open 'sample_grid.png' to see all faces together")
        print("  3. Review 'benchmark_results.csv' for detailed metrics")
        print("  4. If on CPU, consider reducing batch sizes for faster testing")

def quick_test():
    """Quick test to verify everything works and save some images"""
    print("🔧 QUICK TEST - Generate 2 faces to verify setup")
    print("=" * 60)
    
    # Initialize
    benchmark = FFHQBenchmarkFixed()
    
    # Generate just 2 images with minimal steps
    print("\nGenerating 2 test faces...")
    
    shape = [3, 64, 64]
    c = torch.zeros(2, 0, 64, 64).to(benchmark.device)
    uc = torch.zeros(2, 0, 64, 64).to(benchmark.device)
    
    start_time = time.time()
    
    # Use very few steps for quick test
    samples, _ = benchmark.sampler.sample(
        S=10,  # Very few steps for quick test
        conditioning=c,
        batch_size=2,
        shape=shape,
        eta=0.0,
        verbose=False,
        unconditional_guidance_scale=7.5,
        unconditional_conditioning=uc,
    )
    
    with torch.no_grad():
        x_samples = benchmark.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
    
    # Save images
    saved_files = benchmark.save_images(x_samples, 2, 0, step_idx="quick_test")
    
    print(f" Generated 2 faces in {time.time()-start_time:.1f}s")
    print(f" Images saved:")
    for f in saved_files:
        print(f"  • {f}")
    
    # Cleanup
    if os.path.exists(benchmark.config_path):
        os.remove(benchmark.config_path)
    
    return saved_files

def main():
    """Main function"""
    print(" FFHQ BENCHMARK & IMAGE GENERATOR")
    print("=" * 60)
    print("\nOptions:")
    print("1. Quick test (generate 2 images)")
    print("2. Full benchmark (test batch sizes 1, 2, 4)")
    print("3. Just generate sample grid")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    torch.set_grad_enabled(False)
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        # Run full benchmark
        benchmark = FFHQBenchmarkFixed()
        
        # Test small batch sizes (especially on CPU)
        batch_sizes = [1, 2, 4]
        if torch.cuda.is_available():
            batch_sizes = [1, 2, 4, 8]  # Test more on GPU
        
        try:
            results = benchmark.measure_inference_speed(
                batch_sizes=batch_sizes,
                steps=20,  # Reduced for faster testing
                num_repeats=2
            )
            
            # Create visualizations
            benchmark.plot_performance()
            
            # Generate sample grid
            benchmark.create_sample_grid(batch_size=4, steps=30)
            
            # Print report
            benchmark.print_final_report()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup config
            if os.path.exists(benchmark.config_path):
                os.remove(benchmark.config_path)
    
    elif choice == "3":
        # Just generate samples
        benchmark = FFHQBenchmarkFixed()
        try:
            benchmark.create_sample_grid(batch_size=6, steps=40)
        finally:
            if os.path.exists(benchmark.config_path):
                os.remove(benchmark.config_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
