# tsne_thumbnail_visualizer.py
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class TSNEThumbnailVisualizer:
    def __init__(self):
        """Initialize t-SNE thumbnail visualizer"""
        print("🖼️ t-SNE THUMBNAIL VISUALIZATION")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"tsne_thumbnails_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_tsne_results(self, tsne_dir):
        """Load t-SNE results from analysis directory"""
        print(f"\n📂 Loading t-SNE results from: {tsne_dir}")
        
        # Look for the most recent t-SNE analysis or specify one
        analysis_dir = os.path.join(tsne_dir, "analysis")
        visualizations_dir = os.path.join(tsne_dir, "visualizations")
        
        if not os.path.exists(analysis_dir):
            print(f"❌ Analysis directory not found: {analysis_dir}")
            return None
        
        # Load t-SNE summary
        summary_file = os.path.join(analysis_dir, "tsne_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                tsne_summary = json.load(f)
            print(f"✅ Loaded t-SNE summary")
        else:
            print(f"❌ t-SNE summary file not found")
            return None
        
        # Load t-SNE embedding (use selected perplexity)
        selected_perplexity = tsne_summary['selected_perplexity']
        embedding_file = os.path.join(analysis_dir, f"tsne_perplexity{selected_perplexity}.npy")
        
        if os.path.exists(embedding_file):
            tsne_embedding = np.load(embedding_file)
            print(f"✅ Loaded t-SNE embedding (perplexity={selected_perplexity}): {tsne_embedding.shape}")
        else:
            print(f"❌ t-SNE embedding file not found: {embedding_file}")
            return None
        
        return tsne_embedding, tsne_summary
    
    def load_images(self, images_dir):
        """Load original images"""
        print(f"\n📂 Loading images from: {images_dir}")
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(png_files) == 0:
            print("❌ No PNG files found.")
            return None, None
        
        print(f"📊 Found {len(png_files)} images")
        
        # Load images
        all_images = []
        image_arrays = []
        
        for idx, filename in enumerate(png_files):
            img_path = os.path.join(images_dir, filename)
            img = Image.open(img_path)
            
            # Store PIL image
            all_images.append(img)
            
            # Store numpy array for thumbnail
            img_array = np.array(img)
            image_arrays.append(img_array)
        
        return all_images, image_arrays
    
    def create_thumbnail_plot(self, tsne_embedding, images, image_arrays):
        """Create plot with thumbnail images at t-SNE positions"""
        print("\n🖼️ Creating thumbnail visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Normalize t-SNE coordinates for better visualization
        x = tsne_embedding[:, 0]
        y = tsne_embedding[:, 1]
        
        # Add small dots at each position (for reference)
        ax.scatter(x, y, alpha=0.3, s=10, color='gray')
        
        # Add thumbnail images
        print(f"   Adding {len(images)} thumbnails...")
        
        for i, (img_array, xi, yi) in enumerate(zip(image_arrays, x, y)):
            # Create thumbnail (resize to smaller size)
            thumb_size = 64  # pixels
            img_pil = Image.fromarray(img_array)
            img_pil.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            # Convert back to array for matplotlib
            thumb_array = np.array(img_pil)
            
            # Create OffsetImage
            im = OffsetImage(thumb_array, zoom=1.0)
            
            # Create AnnotationBbox
            ab = AnnotationBbox(im, (xi, yi), frameon=True, 
                              pad=0.1, 
                              bboxprops=dict(edgecolor='blue', linewidth=1, alpha=0.7))
            
            # Add to axis
            ax.add_artist(ab)
            
            # Add sample number
            ax.annotate(str(i), (xi, yi), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Set plot properties
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Visualization with Image Thumbnails')
        ax.grid(True, alpha=0.3)
        
        # Adjust limits with some padding
        x_padding = (x.max() - x.min()) * 0.1
        y_padding = (y.max() - y.min()) * 0.1
        ax.set_xlim(x.min() - x_padding, x.max() + x_padding)
        ax.set_ylim(y.min() - y_padding, y.max() + y_padding)
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(self.output_dir, "tsne_thumbnails_full.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved thumbnail plot: {output_file}")
        
        return fig
    
    def create_clustered_thumbnails(self, tsne_embedding, images, image_arrays):
        """Create visualization with thumbnails, colored by cluster"""
        from sklearn.cluster import KMeans
        
        print("\n🎨 Creating clustered thumbnail visualization...")
        
        # Perform clustering
        n_clusters = 4  # You can adjust this
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(tsne_embedding)
        
        # Create colormap for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Thumbnails with cluster colors
        ax1 = axes[0]
        
        # Add thumbnail images with colored borders
        thumb_size = 48
        
        for i, (img_array, xi, yi, label) in enumerate(zip(image_arrays, 
                                                          tsne_embedding[:, 0], 
                                                          tsne_embedding[:, 1], 
                                                          labels)):
            # Create thumbnail
            img_pil = Image.fromarray(img_array)
            img_pil.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            thumb_array = np.array(img_pil)
            
            # Create OffsetImage
            im = OffsetImage(thumb_array, zoom=1.0)
            
            # Create AnnotationBbox with colored border
            ab = AnnotationBbox(im, (xi, yi), frameon=True, 
                              pad=0.1, 
                              bboxprops=dict(edgecolor=colors[label], 
                                           linewidth=2, 
                                           alpha=0.8))
            
            ax1.add_artist(ab)
            
            # Add small cluster number
            ax1.annotate(str(label), (xi, yi), xytext=(2, 2), 
                        textcoords='offset points', fontsize=7,
                        bbox=dict(boxstyle="circle,pad=0.1", facecolor=colors[label], alpha=0.7))
        
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.set_title(f't-SNE with Thumbnails ({n_clusters} Clusters)')
        ax1.grid(True, alpha=0.3)
        
        # Adjust limits
        x = tsne_embedding[:, 0]
        y = tsne_embedding[:, 1]
        x_padding = (x.max() - x.min()) * 0.1
        y_padding = (y.max() - y.min()) * 0.1
        ax1.set_xlim(x.min() - x_padding, x.max() + x_padding)
        ax1.set_ylim(y.min() - y_padding, y.max() + y_padding)
        
        # Plot 2: Cluster collage
        ax2 = axes[1]
        ax2.axis('off')
        
        # Create collage for each cluster
        collage_height = 0
        
        for cluster_id in range(n_clusters):
            # Get images in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            
            if len(cluster_images) == 0:
                continue
            
            # Create mini collage for this cluster
            collage = self.create_cluster_collage(cluster_images, cluster_id)
            
            # Display collage
            ax2.imshow(collage, extent=[0, 10, collage_height, collage_height + 2])
            
            # Add cluster label
            ax2.text(0, collage_height + 1.8, f'Cluster {cluster_id} ({len(cluster_images)} images)', 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[cluster_id], alpha=0.7))
            
            collage_height += 2.5
        
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, collage_height)
        ax2.set_title('Cluster Collages')
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(self.output_dir, "tsne_clustered_thumbnails.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved clustered thumbnail plot: {output_file}")
        
        # Save cluster assignments
        cluster_data = {
            'n_clusters': n_clusters,
            'cluster_assignments': labels.tolist(),
            'cluster_counts': [int(np.sum(labels == i)) for i in range(n_clusters)]
        }
        
        with open(os.path.join(self.output_dir, "cluster_assignments.json"), 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        print(f"✅ Saved cluster assignments: {os.path.join(self.output_dir, 'cluster_assignments.json')}")
        
        return fig, labels
    
    def create_cluster_collage(self, images, cluster_id):
        """Create a collage of images from a cluster"""
        # Create a grid for the collage
        grid_size = min(4, len(images))  # Max 4 images per row
        thumb_size = 100
        
        # Calculate collage dimensions
        n_images = len(images)
        n_rows = (n_images + grid_size - 1) // grid_size
        
        # Create blank canvas
        collage_width = grid_size * thumb_size
        collage_height = n_rows * thumb_size
        collage = Image.new('RGB', (collage_width, collage_height), color='white')
        
        # Paste thumbnails
        for idx, img in enumerate(images):
            row = idx // grid_size
            col = idx % grid_size
            
            # Create thumbnail
            img_copy = img.copy()
            img_copy.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            # Calculate position
            x = col * thumb_size
            y = row * thumb_size
            
            # Paste thumbnail
            collage.paste(img_copy, (x, y))
        
        # Add border
        from PIL import ImageDraw
        draw = ImageDraw.Draw(collage)
        draw.rectangle([0, 0, collage_width-1, collage_height-1], outline='black', width=2)
        
        return np.array(collage)
    
    def create_interactive_html(self, tsne_embedding, images, image_arrays):
        """Create interactive HTML visualization"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            print("\n🌐 Creating interactive HTML visualization...")
            
            # Create figure
            fig = make_subplots(rows=1, cols=1)
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=tsne_embedding[:, 0],
                    y=tsne_embedding[:, 1],
                    mode='markers+text',
                    marker=dict(size=12, color='lightblue', opacity=0.7),
                    text=[str(i) for i in range(len(images))],
                    textposition="top center",
                    hoverinfo='text',
                    name='t-SNE Positions'
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Interactive t-SNE Visualization',
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                hovermode='closest',
                height=800,
                showlegend=False
            )
            
            # Add hover information
            hover_texts = []
            for i in range(len(images)):
                hover_texts.append(f'Sample {i}<br>Position: ({tsne_embedding[i, 0]:.2f}, {tsne_embedding[i, 1]:.2f})')
            
            fig.data[0].hovertext = hover_texts
            
            # Save HTML
            output_file = os.path.join(self.output_dir, "tsne_interactive.html")
            fig.write_html(output_file)
            
            print(f"✅ Saved interactive visualization: {output_file}")
            
            return True
            
        except ImportError:
            print("⚠️ Plotly not installed. Skipping interactive visualization.")
            print("   Install with: pip install plotly")
            return False
    
    def create_image_grid_by_cluster(self, tsne_embedding, images, labels):
        """Create image grids organized by cluster"""
        print("\n🔲 Creating image grids by cluster...")
        
        from sklearn.cluster import KMeans
        
        # If labels not provided, perform clustering
        if labels is None:
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(tsne_embedding)
        else:
            n_clusters = len(np.unique(labels))
        
        # Create a figure for each cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            print(f"   Creating grid for Cluster {cluster_id} ({len(cluster_indices)} images)")
            
            # Determine grid layout
            n_images = len(cluster_indices)
            grid_size = int(np.ceil(np.sqrt(n_images)))
            
            # Create figure
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            
            # Flatten axes if needed
            if grid_size == 1:
                axes = np.array([[axes]])
            axes = axes.flatten()
            
            # Plot images
            for idx, (ax, img_idx) in enumerate(zip(axes, cluster_indices)):
                if idx < n_images:
                    img = images[img_idx]
                    ax.imshow(img)
                    ax.set_title(f'Sample {img_idx}', fontsize=8)
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            # Hide empty subplots
            for idx in range(n_images, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Cluster {cluster_id} - {n_images} images', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"cluster_{cluster_id}_grid.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"     Saved: {output_file}")
    
    def run_visualization(self):
        """Run complete thumbnail visualization pipeline"""
        print(f"\n{'='*60}")
        print("RUNNING THUMBNAIL VISUALIZATION")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Paths to your data
        tsne_dir = "tsne_analysis_20251204_142006"  # Your t-SNE results directory
        images_dir = "latent_analysis_20251202_175108/latents/"  # Your images directory
        
        # Check if directories exist
        if not os.path.exists(tsne_dir):
            # Try to find the most recent t-SNE analysis
            possible_dirs = [d for d in os.listdir('.') if d.startswith('tsne_analysis_')]
            if possible_dirs:
                tsne_dir = sorted(possible_dirs)[-1]  # Use most recent
                print(f"📂 Using t-SNE directory: {tsne_dir}")
            else:
                print("❌ No t-SNE analysis directory found.")
                return None
        
        if not os.path.exists(images_dir):
            # Try alternative paths
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    images_dir = path
                    print(f"📂 Using images directory: {images_dir}")
                    break
        
        # Load t-SNE results
        tsne_data = self.load_tsne_results(tsne_dir)
        if tsne_data is None:
            return None
        
        tsne_embedding, tsne_summary = tsne_data
        
        # Load images
        images, image_arrays = self.load_images(images_dir)
        if images is None:
            return None
        
        print(f"\n✅ Starting visualization with {len(images)} images")
        print(f"   t-SNE embedding shape: {tsne_embedding.shape}")
        print(f"   Selected perplexity: {tsne_summary['selected_perplexity']}")
        
        # Create visualizations
        print(f"\n{'='*40}")
        print("CREATING VISUALIZATIONS")
        print('='*40)
        
        # 1. Basic thumbnail plot
        self.create_thumbnail_plot(tsne_embedding, images, image_arrays)
        
        # 2. Clustered thumbnail visualization
        fig, labels = self.create_clustered_thumbnails(tsne_embedding, images, image_arrays)
        
        # 3. Image grids by cluster
        self.create_image_grid_by_cluster(tsne_embedding, images, labels)
        
        # 4. Interactive HTML (optional)
        self.create_interactive_html(tsne_embedding, images, image_arrays)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        
        print(f"\n📋 Summary:")
        print(f"   Duration: {elapsed_time}")
        print(f"   Images visualized: {len(images)}")
        print(f"   Output directory: {self.output_dir}")
        
        print(f"\n📁 Files Created:")
        print(f"   1. tsne_thumbnails_full.png - Full thumbnail visualization")
        print(f"   2. tsne_clustered_thumbnails.png - Clustered view with collages")
        print(f"   3. cluster_*.png - Image grids for each cluster")
        print(f"   4. cluster_assignments.json - Cluster membership data")
        print(f"   5. tsne_interactive.html - Interactive visualization (if plotly installed)")
        
        print(f"\n🔍 How to Use:")
        print(f"   1. Open tsne_thumbnails_full.png to see all thumbnails at t-SNE positions")
        print(f"   2. Check which images are in each cluster")
        print(f"   3. Look at cluster grids to see visual patterns")
        print(f"   4. Open the interactive HTML for exploration")
        
        return {
            'tsne_embedding': tsne_embedding,
            'images': images,
            'labels': labels,
            'output_dir': self.output_dir,
            'elapsed_time': elapsed_time
        }


def main():
    """Main function"""
    print("\n" + "="*60)
    print("t-SNE THUMBNAIL VISUALIZATION TOOL")
    print("="*60)
    print("\nThis tool creates visualizations with image thumbnails")
    print("placed at their t-SNE coordinates.\n")
    
    # Initialize visualizer
    visualizer = TSNEThumbnailVisualizer()
    
    # Run visualization
    results = visualizer.run_visualization()
    
    if results:
        print(f"\n✅ Visualization completed successfully!")
        print(f"📁 All outputs saved to: {results['output_dir']}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Examine the thumbnails to see if similar images cluster together")
        print(f"   2. Check cluster assignments to understand groupings")
        print(f"   3. Compare with your original images to validate patterns")
    else:
        print("\n❌ Visualization failed or was cancelled.")


if __name__ == "__main__":
    main()
