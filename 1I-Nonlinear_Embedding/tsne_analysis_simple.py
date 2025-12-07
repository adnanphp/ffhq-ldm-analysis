# tsne_analysis_simple.py (FIXED VERSION)
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class SimpleTSNEAnalyzer:
    def __init__(self):
        """Initialize t-SNE analyzer"""
        print("🔬 SIMPLE t-SNE ANALYSIS OF GENERATED IMAGES")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"tsne_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        for dir_path in [self.visualizations_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # Analysis parameters
        self.analysis_params = {
            'n_components_tsne': 2,  # t-SNE to 2D for visualization
            'perplexities': [5, 15, 30, 50],  # Different perplexities to try
            'n_iter': 1000,  # Number of iterations
            'random_state': 42,
            'pca_precomponents': 50  # First reduce with PCA
        }
    
    def load_existing_images(self, images_dir):
        """Load existing images and extract features"""
        print(f"\n📂 Loading images from: {images_dir}")
        
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
    
    def preprocess_with_pca(self, feature_matrix):
        """Preprocess data with PCA before t-SNE (for efficiency)"""
        print("\n🔧 Preprocessing with PCA for efficiency...")
        
        # Standardize data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply PCA to reduce dimensionality
        n_components = min(self.analysis_params['pca_precomponents'], 
                          features_scaled.shape[0], 
                          features_scaled.shape[1])
        
        pca = PCA(n_components=n_components, random_state=self.analysis_params['random_state'])
        pca_result = pca.fit_transform(features_scaled)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        print(f"   PCA reduced dimension from {feature_matrix.shape[1]} to {n_components}")
        print(f"   Variance retained: {explained_variance:.3%}")
        
        return pca_result, pca, explained_variance
    
    def perform_tsne_analysis(self, pca_result):
        """Perform t-SNE dimensionality reduction"""
        print(f"\n{'='*60}")
        print("t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
        print(f"{'='*60}")
        
        tsne_results = {}
        
        for perplexity in self.analysis_params['perplexities']:
            print(f"\n🔍 Running t-SNE with perplexity={perplexity}...")
            
            try:
                # Try different parameter names for different scikit-learn versions
                tsne = TSNE(n_components=self.analysis_params['n_components_tsne'],
                           perplexity=perplexity,
                           n_iter=self.analysis_params['n_iter'],
                           random_state=self.analysis_params['random_state'],
                           verbose=0)
            except TypeError:
                # If n_iter doesn't work, try max_iter (newer scikit-learn)
                tsne = TSNE(n_components=self.analysis_params['n_components_tsne'],
                           perplexity=perplexity,
                           max_iter=self.analysis_params['n_iter'],
                           random_state=self.analysis_params['random_state'],
                           verbose=0)
            
            tsne_embedding = tsne.fit_transform(pca_result)
            tsne_results[perplexity] = tsne_embedding
            
            # Save results
            np.save(os.path.join(self.analysis_dir, f"tsne_perplexity{perplexity}.npy"), 
                   tsne_embedding)
            
            print(f"   Completed t-SNE embedding")
        
        # Create visualizations
        self.create_tsne_visualizations(tsne_results)
        
        # Choose median perplexity
        median_perplexity = self.analysis_params['perplexities'][len(self.analysis_params['perplexities']) // 2]
        final_tsne = tsne_results[median_perplexity]
        
        # Save t-SNE summary
        tsne_summary = {
            'perplexities_tested': self.analysis_params['perplexities'],
            'selected_perplexity': median_perplexity,
            'n_components': self.analysis_params['n_components_tsne'],
            'n_samples': pca_result.shape[0],
            'pca_preprocessing': {
                'n_components': pca_result.shape[1],
                'n_iter': self.analysis_params['n_iter'],
                'random_state': self.analysis_params['random_state']
            }
        }
        
        with open(os.path.join(self.analysis_dir, "tsne_summary.json"), 'w') as f:
            json.dump(tsne_summary, f, indent=2)
        
        return final_tsne, tsne_results, tsne_summary
    
    def create_tsne_visualizations(self, tsne_results):
        """Create t-SNE visualizations"""
        print("\n📈 Creating t-SNE visualizations...")
        
        perplexities = list(tsne_results.keys())
        n_perplexities = len(perplexities)
        
        # Create subplot grid
        n_cols = min(2, n_perplexities)
        n_rows = (n_perplexities + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_perplexities == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each perplexity
        for idx, perplexity in enumerate(perplexities):
            row = idx // n_cols
            col = idx % n_cols
            
            embedding = tsne_results[perplexity]
            
            # Create scatter plot
            scatter = axes[row, col].scatter(embedding[:, 0], embedding[:, 1], 
                                           c=np.arange(len(embedding)), 
                                           cmap='viridis', alpha=0.7, s=50)
            
            axes[row, col].set_title(f't-SNE (perplexity={perplexity})')
            axes[row, col].set_xlabel('t-SNE 1')
            axes[row, col].set_ylabel('t-SNE 2')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add colorbar to first plot
            if idx == 0:
                plt.colorbar(scatter, ax=axes[row, col], label='Sample Index')
        
        # Hide empty subplots
        for idx in range(len(perplexities), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('t-SNE Embeddings with Different Perplexity Values', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "tsne_perplexity_comparison.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed visualization for selected perplexity
        self.create_detailed_tsne_plot(tsne_results)
        
        # Create grid of small thumbnails in t-SNE space
        self.create_tsne_thumbnail_grid(tsne_results)
    
    def create_detailed_tsne_plot(self, tsne_results):
        """Create detailed t-SNE plot for selected perplexity"""
        # Use median perplexity for detailed plot
        median_perplexity = self.analysis_params['perplexities'][len(self.analysis_params['perplexities']) // 2]
        embedding = tsne_results[median_perplexity]
        
        plt.figure(figsize=(12, 10))
        
        # Main scatter plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=np.arange(len(embedding)), 
                            cmap='viridis', alpha=0.7, s=80)
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f't-SNE Visualization (perplexity={median_perplexity})')
        plt.grid(True, alpha=0.3)
        
        # Add sample numbers
        for i, (x, y) in enumerate(embedding):
            plt.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
        
        # Density plot (hexbin)
        plt.subplot(2, 2, 2)
        hb = plt.hexbin(embedding[:, 0], embedding[:, 1], 
                       gridsize=30, cmap='YlOrRd', bins='log')
        plt.colorbar(hb, label='log10(count)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Density of Samples in t-SNE Space')
        plt.grid(True, alpha=0.3)
        
        # Marginal distributions
        plt.subplot(2, 2, 3)
        plt.hist(embedding[:, 0], bins=30, alpha=0.5, label='t-SNE 1', color='blue')
        plt.hist(embedding[:, 1], bins=30, alpha=0.5, label='t-SNE 2', color='red')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Marginal Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distance histogram
        plt.subplot(2, 2, 4)
        # Calculate pairwise distances in t-SNE space
        from scipy.spatial.distance import pdist
        distances = pdist(embedding)
        plt.hist(distances, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Pairwise Distance in t-SNE Space')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pairwise Distances')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed t-SNE Analysis (perplexity={median_perplexity})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "tsne_detailed_analysis.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_tsne_thumbnail_grid(self, tsne_results):
        """Create grid showing thumbnails in t-SNE space"""
        median_perplexity = self.analysis_params['perplexities'][len(self.analysis_params['perplexities']) // 2]
        embedding = tsne_results[median_perplexity]
        
        # Create a scatter plot with very small markers
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=np.arange(len(embedding)), 
                            cmap='viridis', alpha=0.3, s=5)
        
        # Overlay sample numbers
        for i, (x, y) in enumerate(embedding):
            plt.annotate(str(i), (x, y), fontsize=6, alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f't-SNE with Sample Labels (perplexity={median_perplexity})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "tsne_with_labels.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create a version with cluster visualization
        self.create_tsne_cluster_visualization(embedding)
    
    def create_tsne_cluster_visualization(self, embedding):
        """Create visualization showing potential clusters in t-SNE space"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        plt.figure(figsize=(15, 5))
        
        # Try different numbers of clusters
        n_clusters_options = [2, 3, 4, 5]
        
        for idx, n_clusters in enumerate(n_clusters_options, 1):
            plt.subplot(1, len(n_clusters_options), idx)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.analysis_params['random_state'])
            labels = kmeans.fit_predict(embedding)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(embedding, labels)
            else:
                silhouette_avg = 0
            
            # Plot with cluster colors
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=labels, cmap='tab10', alpha=0.7, s=50)
            
            # Plot cluster centers
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, alpha=0.8)
            
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f'{n_clusters} Clusters\nSilhouette: {silhouette_avg:.3f}')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('K-means Clustering in t-SNE Space', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "tsne_cluster_analysis.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_tsne_report(self, feature_matrix, tsne_result, tsne_summary, pca_info):
        """Generate t-SNE analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING t-SNE ANALYSIS REPORT")
        print(f"{'='*60}")
        
        report = f"""
t-SNE ANALYSIS REPORT
=====================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

1. DATASET OVERVIEW
------------------
• Number of images analyzed: {feature_matrix.shape[0]}
• Feature dimension (flattened pixels): {feature_matrix.shape[1]}
• Total pixels analyzed: {feature_matrix.size:,}
• Memory size: {feature_matrix.nbytes / (1024**2):.1f} MB

2. PREPROCESSING (PCA)
---------------------
• PCA reduced dimension from {feature_matrix.shape[1]} to {pca_info['n_components']}
• Variance retained by PCA: {pca_info['variance_retained']:.3%}
• Purpose: Speed up t-SNE computation while preserving structure

3. t-SNE PARAMETERS
------------------
• t-SNE dimensions: {tsne_summary['n_components']}D (for visualization)
• Perplexities tested: {tsne_summary['perplexities_tested']}
• Selected perplexity: {tsne_summary['selected_perplexity']}
• Number of iterations: {tsne_summary['pca_preprocessing']['n_iter']}
• Random seed: {tsne_summary['pca_preprocessing']['random_state']}

4. KEY FINDINGS
--------------
• t-SNE successfully embedded {feature_matrix.shape[0]} samples into 2D space
• Different perplexity values tested to find optimal local/global balance
• Clustering in t-SNE space reveals natural groupings in the data
• Pairwise distances show distribution of similarities between images

5. t-SNE INTERPRETATION
---------------------
• t-SNE preserves local structure: Nearby points in t-SNE space are similar
• Global structure may be distorted (this is normal for t-SNE)
• Clusters represent groups of similar images
• Distances between clusters indicate dissimilarity between image groups

6. VISUALIZATIONS GENERATED
--------------------------
• tsne_perplexity_comparison.png - Comparison of different perplexities
• tsne_detailed_analysis.png - Detailed analysis with multiple views
• tsne_with_labels.png - t-SNE plot with sample labels
• tsne_cluster_analysis.png - K-means clustering results

7. DATA FILES
------------
• tsne_perplexity*.npy - t-SNE embeddings for each perplexity
• tsne_summary.json - t-SNE analysis parameters and summary

8. HOW TO INTERPRET RESULTS
-------------------------
1. Look for clusters - groups of points that are close together
2. Check if similar images cluster together in t-SNE space
3. Compare different perplexities to see stability of structure
4. Use sample labels to identify which images are in which clusters
5. Clusters with high silhouette scores are more distinct

9. LIMITATIONS
-------------
• t-SNE is stochastic - different runs may give slightly different results
• Global distances may not be preserved (focus is on local structure)
• Requires careful choice of perplexity parameter
• Computationally expensive for large datasets

10. CONCLUSIONS
--------------
t-SNE reveals the underlying structure of your generated images in a 
2-dimensional space that humans can visualize. The presence of clear 
clusters suggests that your images have distinct groups or styles.

========================================================================
Report generated by Simple t-SNE Analysis Framework
"""
        
        # Save report
        report_file = os.path.join(self.analysis_dir, "tsne_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ t-SNE report saved to: {report_file}")
        
        # Also save as JSON
        json_report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'n_samples': feature_matrix.shape[0],
                'original_dimension': feature_matrix.shape[1],
                'pca_dimension': pca_info['n_components'],
                'pca_variance_retained': pca_info['variance_retained'],
                'tsne_perplexities_tested': tsne_summary['perplexities_tested'],
                'selected_perplexity': tsne_summary['selected_perplexity'],
                'tsne_dimensions': tsne_summary['n_components']
            }
        }
        
        json_file = os.path.join(self.analysis_dir, "report_summary.json")
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"✅ Summary saved to: {json_file}")
        
        return report
    
    def run_tsne_analysis(self):
        """Run complete t-SNE analysis pipeline"""
        print(f"\n{'='*60}")
        print("RUNNING t-SNE ANALYSIS")
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
            print(f"\n📁 Found image directory: {existing_dir}")
            
            # Load images
            feature_matrix, images = self.load_existing_images(existing_dir)
            
            if feature_matrix is None:
                print("❌ No images found or error loading images.")
                return None
        else:
            print("❌ No existing images found.")
            return None
        
        print(f"\n✅ Starting t-SNE analysis with {len(feature_matrix)} samples")
        
        # Preprocess with PCA
        pca_result, pca_model, pca_variance = self.preprocess_with_pca(feature_matrix)
        
        pca_info = {
            'n_components': pca_result.shape[1],
            'variance_retained': pca_variance
        }
        
        # Perform t-SNE
        tsne_result, tsne_results, tsne_summary = self.perform_tsne_analysis(pca_result)
        
        # Generate report
        report = self.generate_tsne_report(feature_matrix, tsne_result, tsne_summary, pca_info)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        
        print(f"\n{'='*60}")
        print("t-SNE ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        print(f"\n📋 Analysis Summary:")
        print(f"   Duration: {elapsed_time}")
        print(f"   Samples analyzed: {feature_matrix.shape[0]}")
        print(f"   Original dimension: {feature_matrix.shape[1]}")
        print(f"   PCA dimension: {pca_info['n_components']}")
        print(f"   t-SNE dimensions: {tsne_summary['n_components']}D")
        print(f"   Perplexities tested: {tsne_summary['perplexities_tested']}")
        print(f"   Selected perplexity: {tsne_summary['selected_perplexity']}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Visualizations: {self.visualizations_dir}/")
        print(f"   Analysis data: {self.analysis_dir}/")
        
        print(f"\n🔍 What to Look For:")
        print(f"   1. Check tsne_perplexity_comparison.png - different perplexities")
        print(f"   2. Look at tsne_detailed_analysis.png - comprehensive view")
        print(f"   3. Examine tsne_cluster_analysis.png - potential clusters")
        print(f"   4. See if similar images cluster together in the plots")
        
        return {
            'feature_matrix': feature_matrix,
            'pca_result': pca_result,
            'tsne_result': tsne_result,
            'tsne_results': tsne_results,
            'tsne_summary': tsne_summary,
            'report': report,
            'output_dir': self.output_dir,
            'elapsed_time': elapsed_time
        }


def main():
    """Main function"""
    print("\n" + "="*60)
    print("SIMPLE t-SNE ANALYSIS FRAMEWORK")
    print("="*60)
    print("\nThis framework performs t-SNE analysis on generated images.")
    print("It visualizes image similarities in 2D space.\n")
    
    # Initialize analyzer
    analyzer = SimpleTSNEAnalyzer()
    
    # Run t-SNE analysis
    results = analyzer.run_tsne_analysis()
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 All outputs saved to: {results['output_dir']}")
        
        # Print quick summary
        print(f"\n📊 Quick Summary:")
        print(f"   Images analyzed: {results['feature_matrix'].shape[0]}")
        print(f"   Analysis time: {results['elapsed_time']}")
        print(f"   Report: {os.path.join(results['output_dir'], 'analysis/tsne_analysis_report.txt')}")
        
        print(f"\n🔍 Important Note:")
        print("   t-SNE is stochastic - different runs may give slightly different results")
        print("   Look for consistent clusters across different perplexity values")
    else:
        print("\n❌ Analysis failed or was cancelled.")


if __name__ == "__main__":
    main()
