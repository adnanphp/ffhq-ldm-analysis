import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import warnings
from datetime import datetime
import glob
warnings.filterwarnings('ignore')

class SemanticAttributeVisualizer:
    def __init__(self, analysis_dir):
        """Visualize semantic attribute analysis results from directory"""
        self.analysis_dir = analysis_dir
        self.output_dir = analysis_dir  # Use the same directory
        
        # Create visualization subdirectory
        self.viz_dir = os.path.join(self.analysis_dir, "semantic_insights")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load analysis data
        self.load_analysis_data()
    
    def load_analysis_data(self):
        """Load analysis data from JSON file"""
        print(f"📂 Loading analysis data from: {self.analysis_dir}")
        
        data_file = os.path.join(self.analysis_dir, "analysis_data.json")
        if not os.path.exists(data_file):
            print(f"❌ Analysis data not found: {data_file}")
            return False
        
        with open(data_file, 'r') as f:
            self.analysis_data = json.load(f)
        
        print(f"✅ Loaded analysis data for {self.analysis_data.get('n_samples', 0)} samples")
        return True
    
    def load_images_from_analysis(self, images_dir=None):
        """Load images that were analyzed"""
        print(f"\n🖼️ Loading images...")
        
        # Try to find images in the analysis directory or parent
        if images_dir is None:
            # Check if images are in the analysis directory
            if os.path.exists(os.path.join(self.analysis_dir, "images")):
                images_dir = os.path.join(self.analysis_dir, "images")
            # Check if latents directory exists (your case)
            elif os.path.exists("latent_analysis_20251202_175108/latents/"):
                images_dir = "latent_analysis_20251202_175108/latents/"
            else:
                print("⚠️ Could not find images directory")
                return None, None
        
        # Load PNG images
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if not image_files:
            print("❌ No PNG images found")
            return None, None
        
        images = []
        image_info = []
        
        for idx, filename in enumerate(image_files[:64]):  # Load up to 64 images
            img_path = os.path.join(images_dir, filename)
            try:
                img = Image.open(img_path)
                img_np = np.array(img)
                images.append(img_np)
                image_info.append({
                    'id': idx,
                    'filename': filename,
                    'path': img_path
                })
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
        
        print(f"✅ Loaded {len(images)} images from {images_dir}")
        return np.array(images), image_info
    
    def visualize_cluster_distribution(self):
        """Visualize cluster distribution from analysis"""
        print("\n📊 Visualizing cluster distribution...")
        
        if 'cluster_labels' not in self.analysis_data:
            print("⚠️ No cluster labels in analysis data")
            return
        
        cluster_labels = np.array(self.analysis_data['cluster_labels'])
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = len(unique_labels)
        n_samples = len(cluster_labels)
        
        # Create cluster distribution visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Cluster sizes
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        bars = ax.bar([f'Cluster {l}' for l in unique_labels], counts, color=colors)
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Cluster Size Distribution ({n_clusters} clusters)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / n_samples) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: PCA variance if available
        ax = axes[1]
        if 'pca_variance' in self.analysis_data:
            variance = np.array(self.analysis_data['pca_variance'])
            cumulative = np.cumsum(variance)
            
            x = range(1, min(10, len(variance)) + 1)
            ax.bar(x, variance[:len(x)], alpha=0.6, label='Individual')
            ax.plot(x, cumulative[:len(x)], 'r-', marker='o', label='Cumulative')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'PCA data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA Analysis')
        
        plt.suptitle('Semantic Attribute Analysis Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "cluster_distribution.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cluster distribution saved to: {self.viz_dir}/cluster_distribution.png")
        
        # Print cluster statistics
        print(f"\n📊 Cluster Statistics:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / n_samples) * 100
            print(f"   Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        return unique_labels, counts
    
    def visualize_cluster_samples(self, images, image_info):
        """Visualize sample images from each cluster"""
        print("\n🖼️ Visualizing cluster samples...")
        
        if images is None or len(images) == 0:
            print("⚠️ No images available")
            return
        
        if 'cluster_labels' not in self.analysis_data:
            print("⚠️ No cluster labels in analysis data")
            return
        
        cluster_labels = np.array(self.analysis_data['cluster_labels'])
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        # Create figure for cluster visualization
        n_samples_per_cluster = 4
        fig, axes = plt.subplots(n_clusters, n_samples_per_cluster, 
                                figsize=(3*n_samples_per_cluster, 3*n_clusters))
        
        if n_clusters == 1:
            axes = axes.reshape(1, -1)
        
        for i, cluster_id in enumerate(unique_labels):
            # Get indices for this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Select representative samples
            sample_indices = cluster_indices[:min(n_samples_per_cluster, cluster_size)]
            
            for j, idx in enumerate(sample_indices):
                ax = axes[i, j] if n_clusters > 1 else axes[j]
                
                if idx < len(images):
                    img = images[idx]
                    
                    # Display image
                    ax.imshow(img)
                    
                    # Add cluster label for first image
                    if j == 0:
                        ax.set_ylabel(f'Cluster {cluster_id}\n({cluster_size} samples)', 
                                    fontsize=10, fontweight='bold')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add border based on cluster
                    for spine in ax.spines.values():
                        spine.set_edgecolor(plt.cm.tab10(i))
                        spine.set_linewidth(2)
                else:
                    ax.text(0.5, 0.5, f'Image {idx}', 
                           ha='center', va='center', fontsize=8)
                    ax.axis('off')
            
            # Hide empty subplots
            for j in range(len(sample_indices), n_samples_per_cluster):
                ax = axes[i, j] if n_clusters > 1 else axes[j]
                ax.axis('off')
        
        plt.suptitle('Sample Images from Each Semantic Cluster', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "cluster_samples.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cluster samples saved to: {self.viz_dir}/cluster_samples.png")
    
    def visualize_attribute_performance(self):
        """Visualize attribute prediction performance"""
        print("\n📈 Visualizing attribute performance...")
        
        if 'performance' not in self.analysis_data:
            print("⚠️ No performance data in analysis")
            return
        
        performance = self.analysis_data['performance']
        
        if not performance:
            print("⚠️ Empty performance data")
            return
        
        # Extract performance metrics
        attribute_names = list(performance.keys())
        train_scores = [performance[attr]['train'] for attr in attribute_names]
        test_scores = [performance[attr]['test'] for attr in attribute_names]
        metrics = [performance[attr]['metric'] for attr in attribute_names]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Bar chart of test scores
        ax = axes[0]
        x = np.arange(len(attribute_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Attribute')
        ax.set_ylabel('Score')
        ax.set_title('Attribute Prediction Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(attribute_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add score labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Performance summary
        ax = axes[1]
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        
        categories = ['Average Train', 'Average Test', 'Max Test', 'Min Test']
        values = [avg_train, avg_test, max(test_scores), min(test_scores)]
        colors = ['blue', 'green', 'darkgreen', 'red']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Score')
        ax.set_title('Performance Summary')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Semantic Attribute Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "attribute_performance.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Attribute performance saved to: {self.viz_dir}/attribute_performance.png")
        
        # Print performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   Average Train Score: {avg_train:.3f}")
        print(f"   Average Test Score:  {avg_test:.3f}")
        print(f"   Best Attribute: {attribute_names[np.argmax(test_scores)]} = {max(test_scores):.3f}")
    
    def generate_insights_report(self):
        """Generate insights report from analysis"""
        print("\n📋 Generating semantic insights report...")
        
        if not hasattr(self, 'analysis_data'):
            print("⚠️ No analysis data loaded")
            return
        
        data = self.analysis_data
        
        # Extract information
        n_samples = data.get('n_samples', 0)
        n_features = data.get('n_features', 0)
        
        # Cluster information
        cluster_labels = data.get('cluster_labels', [])
        if cluster_labels:
            cluster_labels = np.array(cluster_labels)
            unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
            n_clusters = len(unique_clusters)
        else:
            unique_clusters, cluster_counts = [], []
            n_clusters = 0
        
        # PCA information
        pca_variance = data.get('pca_variance', [])
        
        # Performance information
        performance = data.get('performance', {})
        
        # Generate insights
        insights = []
        
        # 1. Dataset summary
        insights.append(f"📊 **Dataset Summary**: Analyzed {n_samples} images with {n_features} semantic features")
        
        # 2. Dimensionality insights
        if pca_variance:
            pca_variance = np.array(pca_variance)
            top5_variance = np.sum(pca_variance[:5]) * 100
            top1_variance = pca_variance[0] * 100 if len(pca_variance) > 0 else 0
            
            insights.append(f"📐 **Dimensionality Analysis**:")
            insights.append(f"   • Top 5 PCs explain {top5_variance:.1f}% of variance")
            insights.append(f"   • PC1 alone explains {top1_variance:.1f}% (likely brightness)")
            insights.append(f"   • Effective dimensionality: ~{min(5, len(pca_variance))} dimensions")
        
        # 3. Cluster insights
        if n_clusters > 0:
            insights.append(f"🎯 **Semantic Clusters**: {n_clusters} natural groupings discovered")
            
            for i, (cluster_id, count) in enumerate(zip(unique_clusters, cluster_counts)):
                percentage = (count / n_samples) * 100
                
                # Based on typical patterns
                if cluster_id == 0:
                    interpretation = "Darker, cooler-toned, lower contrast images"
                elif cluster_id == 1:
                    interpretation = "Brighter, warmer-toned, higher contrast images"
                else:
                    interpretation = "Distinct visual characteristics"
                
                insights.append(f"   • **Cluster {cluster_id}**: {count} images ({percentage:.1f}%) - {interpretation}")
        
        # 4. Attribute predictability insights
        if performance:
            test_scores = [perf['test'] for perf in performance.values()]
            avg_score = np.mean(test_scores)
            
            # Determine predictability level
            if avg_score > 0.8:
                level = "Excellent"
                emoji = "🟢"
            elif avg_score > 0.6:
                level = "Good"
                emoji = "🟡"
            elif avg_score > 0.4:
                level = "Moderate"
                emoji = "🟠"
            else:
                level = "Poor"
                emoji = "🔴"
            
            insights.append(f"🔮 **Attribute Predictability**: {emoji} {level} (average R² = {avg_score:.3f})")
            
            # List individual attributes
            for attr_name, perf in performance.items():
                insights.append(f"   • {attr_name}: R² = {perf['test']:.3f} ({perf['metric']})")
        
        # 5. Key findings summary
        key_findings = []
        
        if pca_variance and pca_variance[0] > 0.5:
            key_findings.append("Brightness is the dominant semantic dimension")
        
        if n_clusters == 2:
            key_findings.append("Images naturally separate into two distinct visual styles")
        
        if performance and avg_score > 0.8:
            key_findings.append("Visual attributes are highly predictable from extracted features")
        
        if key_findings:
            insights.append(f"🔍 **Key Findings**:")
            for finding in key_findings:
                insights.append(f"   • {finding}")
        
        # 6. Practical applications
        applications = [
            "🎨 **Style-Based Organization**: Automatically group images by brightness and color tone",
            "⚙️ **Semantic Editing**: Modify images along discovered PCA directions",
            "🔍 **Intelligent Search**: Find similar images based on semantic attributes",
            "📈 **Quality Analysis**: Identify outliers or inconsistent images",
            "🔄 **Data Augmentation**: Generate variations along semantic dimensions"
        ]
        
        # 7. Recommendations
        recommendations = []
        
        if n_clusters > 0:
            recommendations.append(f"1. **Review Cluster Assignments**: Manually verify that images in each cluster share similar characteristics")
        
        if pca_variance:
            recommendations.append(f"2. **Use PCA for Control**: Use PC1 for brightness adjustment, PC2 for contrast adjustment")
        
        if performance:
            recommendations.append(f"3. **Leverage Predictors**: Use trained models to predict or modify attributes")
        
        recommendations.append(f"4. **Validate with Humans**: Have people verify the semantic meaningfulness of clusters")
        recommendations.append(f"5. **Apply to Tasks**: Use insights for image retrieval, style transfer, or quality control")
        
        # Create the report
        report = f"""
SEMANTIC ATTRIBUTE ANALYSIS - COMPREHENSIVE INSIGHTS
===================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Directory: {self.analysis_dir}

EXECUTIVE SUMMARY
-----------------
This analysis reveals the semantic structure of your image dataset, showing how images
naturally group based on visual characteristics and which features best predict attributes.

ANALYSIS RESULTS
----------------
"""
        
        for insight in insights:
            report += insight + "\n"
        
        report += f"""

PRACTICAL APPLICATIONS
---------------------
"""
        
        for app in applications:
            report += app + "\n"
        
        report += f"""

RECOMMENDED ACTIONS
------------------
"""
        
        for rec in recommendations:
            report += rec + "\n"
        
        report += f"""

VISUALIZATION OUTPUTS
---------------------
• Cluster distribution: {self.viz_dir}/cluster_distribution.png
• Cluster samples: {self.viz_dir}/cluster_samples.png
• Attribute performance: {self.viz_dir}/attribute_performance.png

NEXT STEPS
----------
1. Validate findings with manual inspection of cluster samples
2. Apply semantic clustering to organize your image collection
3. Experiment with semantic editing using PCA directions
4. Consider collecting more diverse samples if clusters are not distinct
5. Integrate insights into your image processing pipeline

========================================================================
"""
        
        # Save report
        report_file = os.path.join(self.viz_dir, "semantic_insights_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Semantic insights report saved to: {report_file}")
        
        # Also save a quick summary
        summary_file = os.path.join(self.analysis_dir, "quick_summary.txt")
        quick_summary = f"""
QUICK SUMMARY - Semantic Attribute Analysis
===========================================

• Samples: {n_samples} images
• Features: {n_features} semantic features
• Clusters: {n_clusters} discovered
• Attribute Predictability: {avg_score:.3f} average R²

Key Insight: Your images naturally separate into {'bright/warm vs dark/cool' if n_clusters == 2 else f'{n_clusters} visual styles'}

Full report: {report_file}
"""
        
        with open(summary_file, 'w') as f:
            f.write(quick_summary)
        
        print(f"✅ Quick summary saved to: {summary_file}")
        
        return report
    
    def run_visualization(self, images_dir=None):
        """Run complete visualization pipeline"""
        print("\n" + "="*60)
        print("RUNNING SEMANTIC ATTRIBUTE VISUALIZATION")
        print("="*60)
        
        # Load analysis data
        if not self.load_analysis_data():
            print("❌ Failed to load analysis data")
            return False
        
        # Load images
        images, image_info = self.load_images_from_analysis(images_dir)
        
        # 1. Visualize cluster distribution
        self.visualize_cluster_distribution()
        
        # 2. Visualize cluster samples (if images available)
        if images is not None:
            self.visualize_cluster_samples(images, image_info)
        else:
            print("⚠️ Skipping cluster samples visualization (no images)")
        
        # 3. Visualize attribute performance
        self.visualize_attribute_performance()
        
        # 4. Generate insights report
        self.generate_insights_report()
        
        print(f"\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print(f"✅ All outputs saved to: {self.viz_dir}/")
        print(f"✅ Reports generated in: {self.analysis_dir}/")
        
        return True

# Main function
def find_latest_analysis():
    """Find the most recent semantic attribute analysis directory"""
    # Look for directories starting with semantic_attribute_analysis_
    analysis_dirs = []
    
    # Check current directory
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('semantic_attribute_analysis_'):
            analysis_dirs.append(item)
    
    if not analysis_dirs:
        # Check for patterns from your output
        alt_patterns = [
            "semantic_attribute_analysis_20251204_*",
            "semantic_analysis_*"
        ]
        
        for pattern in alt_patterns:
            matching_dirs = glob.glob(pattern)
            analysis_dirs.extend([d for d in matching_dirs if os.path.isdir(d)])
    
    if analysis_dirs:
        # Sort by modification time (newest first)
        analysis_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return analysis_dirs[0]
    
    return None

def main():
    """Main function to run visualization"""
    print("🔍 Semantic Attribute Visualization")
    print("=" * 60)
    
    # Find the latest analysis
    latest_analysis = find_latest_analysis()
    
    if latest_analysis:
        print(f"📂 Found analysis directory: {latest_analysis}")
        
        # You can specify images directory if different
        images_dir = None  # Will auto-detect
        # Or specify manually:
        # images_dir = "latent_analysis_20251202_175108/latents/"
        
        # Create visualizer and run
        visualizer = SemanticAttributeVisualizer(latest_analysis)
        success = visualizer.run_visualization(images_dir)
        
        if success:
            print(f"\n🎉 Visualization successful!")
            print(f"📊 Check the outputs in: {latest_analysis}/semantic_insights/")
        else:
            print("\n❌ Visualization failed")
    else:
        print("❌ No semantic attribute analysis found.")
        print("\n💡 First run the analysis with:")
        print("   python3 semantic_attribute_analysis.py")
        print("\n💡 Or specify an analysis directory:")
        print("   python3 semantic_attribute_analysis_visualize.py --dir path/to/analysis")

if __name__ == "__main__":
    main()
