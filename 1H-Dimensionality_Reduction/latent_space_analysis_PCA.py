# latent_space_analysis_simple.py
import torch
import numpy as np
import os
import sys
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add latent-diffusion to path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

class SimplePCAnalyzer:
    def __init__(self):
        """Initialize PCA analyzer"""
        print(" SIMPLE PCA ANALYSIS OF GENERATED IMAGES")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"pca_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        for dir_path in [self.visualizations_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # Analysis parameters
        self.analysis_params = {
            'n_components_pca': 50,
            'random_state': 42
        }
    
    def load_existing_images(self, images_dir):
        """Load existing images and extract features"""
        print(f"\n Loading images from: {images_dir}")
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(png_files) == 0:
            print("❌ No PNG files found.")
            return None, None
        
        print(f"📊 Found {len(png_files)} images")
        
        # Load images and extract features (pixel values)
        all_features = []
        all_images = []
        
        for idx, filename in enumerate(tqdm(png_files, desc="Loading images")):
            img_path = os.path.join(images_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img) / 255.0  # Normalize to [0, 1]
            
            # Flatten image to use as feature vector
            # Shape: (height, width, channels) -> (height * width * channels)
            feature_vector = img_np.flatten()
            
            all_features.append(feature_vector)
            all_images.append({
                'id': idx,
                'path': img_path,
                'image_np': img_np,
                'filename': filename
            })
        
        feature_matrix = np.array(all_features)
        print(f"✅ Created feature matrix: {feature_matrix.shape}")
        print(f"   Samples: {feature_matrix.shape[0]}")
        print(f"   Features per sample: {feature_matrix.shape[1]}")
        
        return feature_matrix, all_images
    
    def perform_pca_analysis(self, feature_matrix):
        """Perform Principal Component Analysis"""
        print(f"\n{'='*60}")
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print(f"{'='*60}")
        
        # Standardize data (important for PCA)
        print("Standardizing data...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Fit PCA
        print("Fitting PCA...")
        n_components = min(self.analysis_params['n_components_pca'], 
                          features_scaled.shape[0], 
                          features_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=self.analysis_params['random_state'])
        pca_result = pca.fit_transform(features_scaled)
        
        # Analysis results
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"\n PCA Results:")
        print(f"   Original dimension: {feature_matrix.shape[1]}")
        print(f"   Reduced dimension: {pca.n_components_}")
        print(f"   Total variance explained by {pca.n_components_} components: {cumulative_variance[-1]:.4f}")
        print(f"   Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
        print(f"   Components needed for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")
        print(f"   First component explains: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
        print(f"   Second component explains: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
        if len(explained_variance) > 2:
            print(f"   Third component explains: {explained_variance[2]:.4f} ({explained_variance[2]*100:.2f}%)")
        
        # Create visualizations
        self.create_pca_visualizations(pca, pca_result, explained_variance, cumulative_variance)
        
        # Save PCA results
        pca_results = {
            'n_components': pca.n_components_,
            'explained_variance_ratio': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_samples': feature_matrix.shape[0],
            'n_features': feature_matrix.shape[1],
            'components_for_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
            'components_for_99_variance': int(np.argmax(cumulative_variance >= 0.99) + 1)
        }
        
        np.save(os.path.join(self.analysis_dir, "pca_transformed.npy"), pca_result)
        
        with open(os.path.join(self.analysis_dir, "pca_results.json"), 'w') as f:
            json.dump(pca_results, f, indent=2)
        
        return pca_result, pca, pca_results
    
    def create_pca_visualizations(self, pca, pca_result, explained_variance, cumulative_variance):
        """Create PCA visualizations"""
        print("\n📈 Creating PCA visualizations...")
        
        # 1. Scree plot
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 3, 1)
        components = range(1, len(explained_variance) + 1)
        plt.bar(components[:20], explained_variance[:20], alpha=0.7, color='skyblue')
        plt.plot(components[:20], explained_variance[:20], 'r-o', linewidth=2)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot (First 20 PCs)')
        plt.grid(True, alpha=0.3)
        
        # 2. Cumulative variance
        plt.subplot(2, 3, 2)
        plt.plot(components, cumulative_variance, 'g-s', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% variance')
        plt.axhline(y=0.99, color='b', linestyle='--', alpha=0.5, label='99% variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. PCA 2D scatter plot
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=np.arange(len(pca_result)), 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
        plt.title('PCA: PC1 vs PC2')
        plt.grid(True, alpha=0.3)
        
        # 4. PCA 3D scatter plot (if we have at least 3 components)
        ax = plt.subplot(2, 3, 4, projection='3d')
        if pca_result.shape[1] >= 3:
            scatter3d = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                                  c=np.arange(len(pca_result)), 
                                  cmap='plasma', alpha=0.6, s=50)
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
            ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.2f}%)')
            ax.set_title('PCA: First 3 Components')
        else:
            ax.text(0.5, 0.5, 0.5, '3D plot not available\n(not enough components)',
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title('PCA: 3D Plot')
        
        # 5. Heatmap of first few components
        plt.subplot(2, 3, 5)
        components_to_show = min(10, pca.components_.shape[0])
        im = plt.imshow(pca.components_[:components_to_show], aspect='auto', 
                       cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        plt.colorbar(im)
        plt.xlabel('Original Feature (pixel index)')
        plt.ylabel('Principal Component')
        plt.title(f'First {components_to_show} PCA Components')
        
        # 6. Distribution along PC1
        plt.subplot(2, 3, 6)
        plt.hist(pca_result[:, 0], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('PC1 Value')
        plt.ylabel('Frequency')
        plt.title('Distribution along PC1')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('PCA Analysis of Generated FFHQ Images', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "pca_analysis.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create additional detailed visualizations
        self.create_detailed_pca_plots(pca_result, explained_variance, cumulative_variance)
    
    def create_detailed_pca_plots(self, pca_result, explained_variance, cumulative_variance):
        """Create additional detailed PCA plots"""
        
        # 1. Variance explained by each component (full plot)
        plt.figure(figsize=(10, 6))
        components = range(1, len(explained_variance) + 1)
        
        plt.subplot(1, 2, 1)
        plt.bar(components, explained_variance, alpha=0.7, color='skyblue')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Variance Explained by Each Component')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(components, cumulative_variance, 'g-s', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% variance')
        plt.axhline(y=0.99, color='b', linestyle='--', alpha=0.5, label='99% variance')
        plt.fill_between(components, 0, cumulative_variance, alpha=0.3, color='green')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "pca_variance_detailed.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. PC1 vs PC2 with sample labels
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=np.arange(len(pca_result)), 
                            cmap='viridis', alpha=0.7, s=100)
        
        # Add sample numbers
        for i in range(len(pca_result)):
            plt.annotate(str(i), (pca_result[i, 0], pca_result[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
        plt.title('PCA: PC1 vs PC2 with Sample Labels')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.visualizations_dir, "pca_scatter_with_labels.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Biplot (PC1 vs PC2 with loadings)
        if pca_result.shape[1] >= 2:
            self.create_biplot(pca_result, explained_variance)
    
    def create_biplot(self, pca_result, explained_variance):
        """Create biplot showing both samples and variable loadings"""
        try:
            # We'll create a simplified biplot showing the direction of maximum variance
            plt.figure(figsize=(10, 8))
            
            # Plot samples
            plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                       c=np.arange(len(pca_result)), 
                       cmap='viridis', alpha=0.6, s=50)
            
            # Add arrows for PC directions (simplified)
            # In a real biplot, these would show variable loadings
            plt.arrow(0, 0, 2*np.std(pca_result[:, 0]), 0, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', label='PC1 Direction')
            plt.arrow(0, 0, 0, 2*np.std(pca_result[:, 1]), 
                     head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='PC2 Direction')
            
            plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
            plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
            plt.title('PCA Biplot (Samples and Principal Directions)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            plt.savefig(os.path.join(self.visualizations_dir, "pca_biplot.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f" Could not create biplot: {e}")
    
    def generate_pca_report(self, feature_matrix, pca_result, pca_results):
        """Generate PCA analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING PCA ANALYSIS REPORT")
        print(f"{'='*60}")

        report = f"PCA Report generated at {datetime.now()}"

        
        # Save report
        report_file = os.path.join(self.analysis_dir, "pca_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ PCA report saved to: {report_file}")
        
        # Also save as JSON
        json_report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'n_samples': feature_matrix.shape[0],
                'original_dimension': feature_matrix.shape[1],
                'pca_dimension': pca_results['n_components'],
                'variance_explained': pca_results['cumulative_variance'][-1],
                'components_for_95_variance': pca_results['components_for_95_variance'],
                'components_for_99_variance': pca_results['components_for_99_variance'],
                'variance_pc1': pca_results['explained_variance_ratio'][0],
                'variance_pc2': pca_results['explained_variance_ratio'][1]
            }
        }
        
        json_file = os.path.join(self.analysis_dir, "report_summary.json")
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"✅ Summary saved to: {json_file}")
        
        return report
    
    def run_pca_analysis(self):
        """Run complete PCA analysis pipeline"""
        print(f"\n{'='*60}")
        print("RUNNING PCA ANALYSIS")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Check for existing images
        existing_dir = "latent_analysis_20251202_175108/latents/"
        
        if not os.path.exists(existing_dir):
            # Try alternative paths
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
                os.path.join(os.getcwd(), "latent_analysis_20251202_175108", "latents"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    existing_dir = path
                    print(f"✅ Found images at: {path}")
                    break
        
        if os.path.exists(existing_dir):
            print(f"\n Found image directory: {existing_dir}")
            
            # Load images
            feature_matrix, images = self.load_existing_images(existing_dir)
            
            if feature_matrix is None:
                print("❌ No images found or error loading images.")
                return None
        else:
            print("❌ No existing images found.")
            return None
        
        print(f"\n✅ Starting PCA analysis with {len(feature_matrix)} samples")
        
        # Perform PCA
        pca_result, pca_model, pca_results = self.perform_pca_analysis(feature_matrix)
        
        # Generate report
        report = self.generate_pca_report(feature_matrix, pca_result, pca_results)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        
        print(f"\n{'='*60}")
        print("PCA ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        print(f"\n Analysis Summary:")
        print(f"   Duration: {elapsed_time}")
        print(f"   Samples analyzed: {feature_matrix.shape[0]}")
        print(f"   Original dimension: {feature_matrix.shape[1]}")
        print(f"   PCA dimension: {pca_results['n_components']}")
        print(f"   Variance explained: {pca_results['cumulative_variance'][-1]:.3%}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Visualizations: {self.visualizations_dir}/")
        print(f"   Analysis data: {self.analysis_dir}/")
        
        print(f"\n🔍 Key Findings:")
        print(f"   • 1st PC explains: {pca_results['explained_variance_ratio'][0]*100:.2f}%")
        print(f"   • 2nd PC explains: {pca_results['explained_variance_ratio'][1]*100:.2f}%")
        print(f"   • Need {pca_results['components_for_95_variance']} PCs for 95% variance")
        
        return {
            'feature_matrix': feature_matrix,
            'pca_result': pca_result,
            'pca_results': pca_results,
            'report': report,
            'output_dir': self.output_dir,
            'elapsed_time': elapsed_time
        }


def main():
    """Main function"""
    print("\n" + "="*60)
    print("SIMPLE PCA ANALYSIS FRAMEWORK")
    print("="*60)
    print("\nThis framework performs PCA analysis on generated images.")
    print("It analyzes the pixel space structure of your images.\n")
    
    # Initialize analyzer
    analyzer = SimplePCAnalyzer()
    
    # Run PCA analysis
    results = analyzer.run_pca_analysis()
    
    if results:
        print(f"\n Analysis completed successfully!")
        print(f" All outputs saved to: {results['output_dir']}")
        
        # Print quick summary
        print(f"\n Quick Summary:")
        print(f"   Images analyzed: {results['feature_matrix'].shape[0]}")
        print(f"   Analysis time: {results['elapsed_time']}")
        print(f"   Report: {os.path.join(results['output_dir'], 'analysis/pca_analysis_report.txt')}")
    else:
        print("\n Analysis failed or was cancelled.")


if __name__ == "__main__":
    main()
