# cluster_analysis_detailed.py
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

class DetailedClusterAnalyzer:
    def __init__(self):
        """Initialize detailed cluster analyzer"""
        print(" DETAILED CLUSTER ANALYSIS")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"cluster_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        self.clusters_dir = os.path.join(self.output_dir, "clusters")
        
        for dir_path in [self.visualizations_dir, self.analysis_dir, self.clusters_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f" Output directory: {self.output_dir}")
        
        # Analysis parameters
        self.analysis_params = {
            'random_state': 42,
            'max_clusters': 10,  # Maximum number of clusters to try
            'min_clusters': 2,   # Minimum number of clusters to try
        }
    
    def load_images_and_features(self, images_dir):
        """Load images and extract features"""
        print(f"\n Loading images from: {images_dir}")
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(png_files) == 0:
            print("❌ No PNG files found.")
            return None, None, None
        
        print(f"📊 Found {len(png_files)} images")
        
        # Load images and extract features
        all_images = []
        all_features = []
        image_info = []
        
        for idx, filename in enumerate(tqdm(png_files, desc="Loading images")):
            img_path = os.path.join(images_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img) / 255.0  # Normalize to [0, 1]
            
            # Store image
            all_images.append(img)
            
            # Extract multiple feature types
            features = self.extract_image_features(img_np)
            all_features.append(features)
            
            image_info.append({
                'id': idx,
                'path': img_path,
                'filename': filename,
                'image_np': img_np
            })
        
        feature_matrix = np.array(all_features)
        print(f"✅ Created feature matrix: {feature_matrix.shape}")
        print(f"   Samples: {feature_matrix.shape[0]}")
        print(f"   Features per sample: {feature_matrix.shape[1]}")
        
        return feature_matrix, all_images, image_info
    
    def extract_image_features(self, image_np):
        """Extract multiple types of features from an image"""
        features = []
        
        # 1. Color features (mean and std of each channel)
        if len(image_np.shape) == 3:  # RGB image
            for channel in range(3):
                channel_data = image_np[:, :, channel].flatten()
                features.extend([np.mean(channel_data), np.std(channel_data), 
                               np.percentile(channel_data, 25), np.percentile(channel_data, 75)])
        
        # 2. Texture features (simplified - using gradient magnitude)
        if len(image_np.shape) == 3:
            gray = np.mean(image_np, axis=2)  # Convert to grayscale
        else:
            gray = image_np
        
        # Simple gradient features
        from scipy.ndimage import sobel
        grad_x = sobel(gray, axis=0)
        grad_y = sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([np.mean(gradient_magnitude), np.std(gradient_magnitude),
                        np.max(gradient_magnitude), np.percentile(gradient_magnitude, 90)])
        
        # 3. Shape/Edge features (using edge density)
        from scipy.ndimage import gaussian_gradient_magnitude
        edges = gaussian_gradient_magnitude(gray, sigma=1)
        edge_density = np.mean(edges > np.percentile(edges, 50))
        features.append(edge_density)
        
        # 4. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])
        
        # 5. Color histogram moments (simplified)
        if len(image_np.shape) == 3:
            for channel in range(3):
                hist, _ = np.histogram(image_np[:, :, channel].flatten(), bins=10)
                hist = hist / hist.sum()  # Normalize
                features.extend([np.mean(hist), np.std(hist), np.argmax(hist)/10])
        
        return np.array(features)
    
    def determine_optimal_clusters(self, feature_matrix):
        """Determine optimal number of clusters using multiple methods"""
        print(f"\n{'='*60}")
        print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
        print(f"{'='*60}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Reduce dimensionality for some methods
        pca = PCA(n_components=min(20, features_scaled.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        
        results = []
        n_clusters_range = range(self.analysis_params['min_clusters'], 
                                self.analysis_params['max_clusters'] + 1)
        
        for n_clusters in tqdm(n_clusters_range, desc="Testing cluster counts"):
            cluster_metrics = {'n_clusters': n_clusters}
            
            # 1. KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.analysis_params['random_state'])
            kmeans_labels = kmeans.fit_predict(features_scaled)
            
            if len(np.unique(kmeans_labels)) > 1:
                cluster_metrics['kmeans_silhouette'] = silhouette_score(features_scaled, kmeans_labels)
                cluster_metrics['kmeans_calinski'] = calinski_harabasz_score(features_scaled, kmeans_labels)
                cluster_metrics['kmeans_davies'] = davies_bouldin_score(features_scaled, kmeans_labels)
                cluster_metrics['kmeans_inertia'] = kmeans.inertia_
            else:
                cluster_metrics['kmeans_silhouette'] = np.nan
                cluster_metrics['kmeans_calinski'] = np.nan
                cluster_metrics['kmeans_davies'] = np.nan
                cluster_metrics['kmeans_inertia'] = np.nan
            
            # 2. Gaussian Mixture (on PCA-reduced data)
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=self.analysis_params['random_state'])
                gmm_labels = gmm.fit_predict(features_pca)
                
                if len(np.unique(gmm_labels)) > 1:
                    cluster_metrics['gmm_silhouette'] = silhouette_score(features_pca, gmm_labels)
                    cluster_metrics['gmm_bic'] = gmm.bic(features_pca)
                    cluster_metrics['gmm_aic'] = gmm.aic(features_pca)
                else:
                    cluster_metrics['gmm_silhouette'] = np.nan
                    cluster_metrics['gmm_bic'] = np.nan
                    cluster_metrics['gmm_aic'] = np.nan
            except:
                cluster_metrics['gmm_silhouette'] = np.nan
                cluster_metrics['gmm_bic'] = np.nan
                cluster_metrics['gmm_aic'] = np.nan
            
            results.append(cluster_metrics)
        
        # Find optimal number of clusters for each method
        optimal_results = self.find_optimal_clusters(results)
        
        # Create visualizations
        self.plot_cluster_metrics(results, optimal_results)
        
        return optimal_results, results
    
    def find_optimal_clusters(self, results):
        """Find optimal number of clusters from metrics"""
        optimal = {}
        
        # Extract metrics
        n_clusters = [r['n_clusters'] for r in results]
        kmeans_silhouette = [r['kmeans_silhouette'] for r in results]
        kmeans_inertia = [r['kmeans_inertia'] for r in results]
        gmm_bic = [r['gmm_bic'] for r in results]
        gmm_aic = [r['gmm_aic'] for r in results]
        
        # Find optimal based on silhouette score (higher is better)
        if not all(np.isnan(kmeans_silhouette)):
            optimal['by_silhouette'] = n_clusters[np.nanargmax(kmeans_silhouette)]
        else:
            optimal['by_silhouette'] = None
        
        # Find elbow point for inertia
        if not all(np.isnan(kmeans_inertia)):
            # Simple elbow detection (second derivative)
            diffs = np.diff(kmeans_inertia)
            diffs2 = np.diff(diffs)
            if len(diffs2) > 0:
                elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
                optimal['by_elbow'] = n_clusters[min(elbow_idx, len(n_clusters)-1)]
            else:
                optimal['by_elbow'] = n_clusters[2]  # Default to 3 clusters
        else:
            optimal['by_elbow'] = None
        
        # Find optimal by BIC (lower is better)
        if not all(np.isnan(gmm_bic)):
            optimal['by_bic'] = n_clusters[np.nanargmin(gmm_bic)]
        else:
            optimal['by_bic'] = None
        
        # Find optimal by AIC (lower is better)
        if not all(np.isnan(gmm_aic)):
            optimal['by_aic'] = n_clusters[np.nanargmin(gmm_aic)]
        else:
            optimal['by_aic'] = None
        
        # Final recommendation (prioritize silhouette if available)
        if optimal['by_silhouette']:
            optimal['recommended'] = optimal['by_silhouette']
        elif optimal['by_elbow']:
            optimal['recommended'] = optimal['by_elbow']
        elif optimal['by_bic']:
            optimal['recommended'] = optimal['by_bic']
        else:
            optimal['recommended'] = 4  # Default
        
        print(f"\n Optimal Cluster Recommendations:")
        print(f"   • By silhouette score: {optimal['by_silhouette']}")
        print(f"   • By elbow method: {optimal['by_elbow']}")
        print(f"   • By BIC: {optimal['by_bic']}")
        print(f"   • By AIC: {optimal['by_aic']}")
        print(f"   ⭐ Recommended: {optimal['recommended']} clusters")
        
        return optimal
    
    def plot_cluster_metrics(self, results, optimal_results):
        """Plot cluster evaluation metrics"""
        print("\n Creating cluster metric visualizations...")
        
        n_clusters = [r['n_clusters'] for r in results]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Silhouette scores
        ax1 = axes[0, 0]
        kmeans_silhouette = [r['kmeans_silhouette'] for r in results]
        gmm_silhouette = [r['gmm_silhouette'] for r in results]
        
        ax1.plot(n_clusters, kmeans_silhouette, 'b-o', label='KMeans', linewidth=2)
        ax1.plot(n_clusters, gmm_silhouette, 'r-s', label='GMM', linewidth=2)
        
        if optimal_results['by_silhouette']:
            ax1.axvline(x=optimal_results['by_silhouette'], color='g', linestyle='--', 
                       label=f'Optimal: {optimal_results["by_silhouette"]}')
        
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score (Higher is Better)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Inertia (Elbow method)
        ax2 = axes[0, 1]
        inertia = [r['kmeans_inertia'] for r in results]
        ax2.plot(n_clusters, inertia, 'g-^', linewidth=2)
        
        if optimal_results['by_elbow']:
            ax2.axvline(x=optimal_results['by_elbow'], color='r', linestyle='--',
                       label=f'Elbow: {optimal_results["by_elbow"]}')
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Inertia')
        ax2.set_title('KMeans Inertia (Elbow Method)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. BIC and AIC
        ax3 = axes[0, 2]
        bic = [r['gmm_bic'] for r in results]
        aic = [r['gmm_aic'] for r in results]
        
        ax3.plot(n_clusters, bic, 'm-d', label='BIC', linewidth=2)
        ax3.plot(n_clusters, aic, 'c-*', label='AIC', linewidth=2)
        
        if optimal_results['by_bic']:
            ax3.axvline(x=optimal_results['by_bic'], color='b', linestyle='--',
                       label=f'BIC optimal: {optimal_results["by_bic"]}')
        
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Score')
        ax3.set_title('GMM BIC/AIC (Lower is Better)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Calinski-Harabasz Index
        ax4 = axes[1, 0]
        calinski = [r['kmeans_calinski'] for r in results]
        ax4.plot(n_clusters, calinski, 'orange', marker='o', linewidth=2)
        ax4.set_xlabel('Number of Clusters')
        ax4.set_ylabel('Calinski-Harabasz Index')
        ax4.set_title('Calinski-Harabasz (Higher is Better)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Davies-Bouldin Index
        ax5 = axes[1, 1]
        davies = [r['kmeans_davies'] for r in results]
        ax5.plot(n_clusters, davies, 'purple', marker='s', linewidth=2)
        ax5.set_xlabel('Number of Clusters')
        ax5.set_ylabel('Davies-Bouldin Index')
        ax5.set_title('Davies-Bouldin (Lower is Better)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary table
        ax6 = axes[1, 2]
        ax6.axis('tight')
        ax6.axis('off')
        
        summary_text = f"Optimal Cluster Analysis\n\n"
        summary_text += f"Recommended: {optimal_results['recommended']} clusters\n\n"
        summary_text += f"By silhouette: {optimal_results['by_silhouette']}\n"
        summary_text += f"By elbow: {optimal_results['by_elbow']}\n"
        summary_text += f"By BIC: {optimal_results['by_bic']}\n"
        summary_text += f"By AIC: {optimal_results['by_aic']}"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Cluster Evaluation Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "cluster_metrics_comparison.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create hierarchical clustering dendrogram
        self.create_dendrogram(results[0]['n_clusters'])  # Using first cluster count as example
    
    def create_dendrogram(self, max_clusters):
        """Create hierarchical clustering dendrogram"""
        # This is a simplified version - in practice, you'd use actual data
        plt.figure(figsize=(12, 8))
        
        # Create example data for dendrogram
        np.random.seed(self.analysis_params['random_state'])
        example_data = np.random.randn(20, 5)
        
        # Perform hierarchical clustering
        linked = linkage(example_data, 'ward')
        
        # Create dendrogram
        dendrogram(linked, orientation='top', 
                  labels=[f'Sample {i}' for i in range(20)],
                  distance_sort='descending',
                  show_leaf_counts=True)
        
        plt.title('Hierarchical Clustering Dendrogram (Example)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        # Add cluster cut lines
        for n_clusters in [2, 3, 4, 5]:
            plt.axhline(y=linked[-(n_clusters-1), 2], color='r', linestyle='--', 
                       alpha=0.5, label=f'{n_clusters} clusters' if n_clusters == 2 else '')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "dendrogram_example.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def perform_clustering(self, feature_matrix, optimal_n_clusters):
        """Perform clustering with optimal number of clusters"""
        print(f"\n{'='*60}")
        print(f"PERFORMING CLUSTERING (k={optimal_n_clusters})")
        print(f"{'='*60}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Try different clustering algorithms
        clustering_results = {}
        
        # 1. KMeans
        print("\n Running KMeans clustering...")
        kmeans = KMeans(n_clusters=optimal_n_clusters, 
                       random_state=self.analysis_params['random_state'])
        kmeans_labels = kmeans.fit_predict(features_scaled)
        clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
        
        # 2. Agglomerative Clustering
        print(" Running Agglomerative clustering...")
        agg = AgglomerativeClustering(n_clusters=optimal_n_clusters)
        agg_labels = agg.fit_predict(features_scaled)
        clustering_results['agglomerative'] = {
            'labels': agg_labels
        }
        
        # 3. DBSCAN (automatically determines clusters)
        print("🔍 Running DBSCAN clustering...")
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            dbscan_labels = dbscan.fit_predict(features_scaled)
            n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
            clustering_results['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters_found': n_dbscan_clusters
            }
            print(f"   DBSCAN found {n_dbscan_clusters} clusters")
        except Exception as e:
            print(f"   DBSCAN failed: {e}")
            clustering_results['dbscan'] = None
        
        # 4. Gaussian Mixture Model
        print("🔍 Running Gaussian Mixture Model...")
        gmm = GaussianMixture(n_components=optimal_n_clusters, 
                             random_state=self.analysis_params['random_state'])
        gmm_labels = gmm.fit_predict(features_scaled)
        clustering_results['gmm'] = {
            'labels': gmm_labels,
            'means': gmm.means_,
            'covariances': gmm.covariances_
        }
        
        # Calculate metrics for each method
        for method, result in clustering_results.items():
            if result is not None and 'labels' in result:
                labels = result['labels']
                if len(np.unique(labels)) > 1:
                    result['silhouette'] = silhouette_score(features_scaled, labels)
                    result['calinski'] = calinski_harabasz_score(features_scaled, labels)
                    result['davies'] = davies_bouldin_score(features_scaled, labels)
        
        # Visualize clustering results
        self.visualize_clustering_results(features_2d, clustering_results, optimal_n_clusters)
        
        return clustering_results, features_2d
    
    def visualize_clustering_results(self, features_2d, clustering_results, n_clusters):
        """Visualize results from different clustering algorithms"""
        print("\n Creating clustering visualizations...")
        
        methods = list(clustering_results.keys())
        n_methods = len(methods)
        
        # Create subplot grid
        n_cols = 2
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_methods == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each method
        for idx, method in enumerate(methods):
            row = idx // n_cols
            col = idx % n_cols
            
            if clustering_results[method] is None:
                axes[row, col].text(0.5, 0.5, f'{method}\nFailed', 
                                   horizontalalignment='center',
                                   verticalalignment='center')
                axes[row, col].set_title(f'{method}')
                continue
            
            labels = clustering_results[method]['labels']
            
            # Create scatter plot
            scatter = axes[row, col].scatter(features_2d[:, 0], features_2d[:, 1], 
                                           c=labels, cmap='tab10', alpha=0.7, s=50)
            
            # Add metrics to title
            title = method.capitalize()
            if 'silhouette' in clustering_results[method]:
                title += f'\nSilhouette: {clustering_results[method]["silhouette"]:.3f}'
            
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('PC1')
            axes[row, col].set_ylabel('PC2')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[row, col], label='Cluster')
        
        # Hide empty subplots
        for idx in range(len(methods), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Clustering Results Comparison (k={n_clusters})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "clustering_comparison.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed KMeans visualization
        if 'kmeans' in clustering_results and clustering_results['kmeans'] is not None:
            self.create_detailed_kmeans_visualization(features_2d, clustering_results['kmeans'])
    
    def create_detailed_kmeans_visualization(self, features_2d, kmeans_results):
        """Create detailed visualization of KMeans results"""
        labels = kmeans_results['labels']
        centers_2d = PCA(n_components=2).fit_transform(kmeans_results['centers'])
        
        plt.figure(figsize=(15, 5))
        
        # 1. Cluster visualization with centers
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=50)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='X', s=200, alpha=0.8, label='Centroids')
        
        # Add Voronoi regions
        from scipy.spatial import Voronoi, voronoi_plot_2d
        try:
            vor = Voronoi(centers_2d)
            voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False, 
                          line_colors='orange', line_width=1, line_alpha=0.3)
        except:
            pass
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('KMeans Clusters with Centroids')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Cluster sizes
        plt.subplot(1, 3, 2)
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        bars = plt.bar(unique_labels.astype(str), counts, color=colors)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Images')
        plt.title('Cluster Sizes')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        # 3. Silhouette plot
        plt.subplot(1, 3, 3)
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(features_2d, labels)
        
        y_lower = 10
        for i in range(len(unique_labels)):
            ith_cluster_silhouette_vals = silhouette_vals[labels == i]
            ith_cluster_silhouette_vals.sort()
            
            size_cluster_i = ith_cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(float(i) / len(unique_labels))
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster label")
        plt.title("Silhouette Plot for Clusters")
        plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
        plt.text(np.mean(silhouette_vals) + 0.01, y_lower - 30,
                f"Average: {np.mean(silhouette_vals):.3f}", color="red")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "kmeans_detailed_analysis.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_cluster_characteristics(self, images, image_info, labels, feature_matrix):
        """Analyze and describe cluster characteristics"""
        print(f"\n{'='*60}")
        print("ANALYZING CLUSTER CHARACTERISTICS")
        print(f"{'='*60}")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        cluster_analysis = {}
        
        for cluster_id in unique_labels:
            print(f"\n Analyzing Cluster {cluster_id}...")
            
            # Get indices of images in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            cluster_features = feature_matrix[cluster_indices]
            
            # Basic statistics
            cluster_size = len(cluster_indices)
            
            # Calculate average features for this cluster
            avg_features = np.mean(cluster_features, axis=0)
            
            # Analyze image characteristics
            characteristics = self.analyze_image_cluster(cluster_images, cluster_features)
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'indices': cluster_indices.tolist(),
                'image_filenames': [image_info[i]['filename'] for i in cluster_indices],
                'average_features': avg_features.tolist(),
                'characteristics': characteristics
            }
            
            # Print summary
            print(f"   Size: {cluster_size} images")
            print(f"   Characteristics: {characteristics}")
            
            # Save cluster images
            self.save_cluster_images(cluster_id, cluster_images, cluster_indices, image_info)
        
        # Compare clusters
        self.compare_clusters(cluster_analysis, feature_matrix, labels)
        
        return cluster_analysis
    
    def analyze_image_cluster(self, cluster_images, cluster_features):
        """Analyze visual characteristics of a cluster"""
        characteristics = []
        
        if len(cluster_images) == 0:
            return ["Empty cluster"]
        
        # Calculate average brightness from features
        # Assuming brightness is at position -2 in our feature vector
        avg_brightness = np.mean(cluster_features[:, -2])
        
        # Simple categorization
        if avg_brightness > 0.6:
            characteristics.append("Bright")
        elif avg_brightness < 0.4:
            characteristics.append("Dark")
        else:
            characteristics.append("Medium brightness")
        
        # Analyze color (simplified)
        # Assuming color features are at positions 0-11
        if cluster_features.shape[1] >= 12:
            color_features = cluster_features[:, :12]
            color_variation = np.std(color_features)
            if color_variation > 0.1:
                characteristics.append("Colorful")
            else:
                characteristics.append("Monochromatic")
        
        # Analyze texture
        if cluster_features.shape[1] >= 16:
            texture_features = cluster_features[:, 12:16]
            avg_texture = np.mean(texture_features)
            if avg_texture > 0.05:
                characteristics.append("Textured")
            else:
                characteristics.append("Smooth")
        
        return characteristics
    
    def save_cluster_images(self, cluster_id, cluster_images, cluster_indices, image_info):
        """Save images from a cluster to a directory"""
        cluster_dir = os.path.join(self.clusters_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        for idx, (img, img_idx) in enumerate(zip(cluster_images, cluster_indices)):
            # Save individual image
            img_filename = f"sample_{img_idx:03d}_{image_info[img_idx]['filename']}"
            img.save(os.path.join(cluster_dir, img_filename))
        
        # Create collage
        self.create_cluster_collage(cluster_id, cluster_images, cluster_indices)
    
    def create_cluster_collage(self, cluster_id, cluster_images, cluster_indices):
        """Create a collage of all images in a cluster"""
        if len(cluster_images) == 0:
            return
        
        # Determine grid size
        n_images = len(cluster_images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        thumb_size = 150
        
        # Create blank canvas
        collage_width = grid_size * thumb_size
        collage_height = grid_size * thumb_size
        collage = Image.new('RGB', (collage_width, collage_height), color='white')
        
        # Paste thumbnails
        for idx, img in enumerate(cluster_images):
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
            
            # Add sample number
            draw = ImageDraw.Draw(collage)
            sample_num = cluster_indices[idx]
            draw.text((x + 5, y + 5), str(sample_num), fill='white', stroke_width=2, stroke_fill='black')
        
        # Add border and title
        draw = ImageDraw.Draw(collage)
        draw.rectangle([0, 0, collage_width-1, collage_height-1], outline='black', width=3)
        draw.text((10, collage_height - 30), f"Cluster {cluster_id} - {n_images} images", 
                 fill='black', stroke_width=1, stroke_fill='white')
        
        # Save collage
        collage.save(os.path.join(self.clusters_dir, f"cluster_{cluster_id}_collage.png"))
    
    def compare_clusters(self, cluster_analysis, feature_matrix, labels):
        """Compare clusters and create visualization"""
        print("\n Creating cluster comparison visualization...")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Create heatmap of average features per cluster
        avg_features_matrix = np.zeros((n_clusters, feature_matrix.shape[1]))
        
        for i, cluster_id in enumerate(unique_labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            avg_features_matrix[i] = np.mean(feature_matrix[cluster_indices], axis=0)
        
        # Normalize for visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(avg_features_matrix.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create subplot for heatmap
        plt.subplot(2, 1, 1)
        im = plt.imshow(normalized_features, aspect='auto', cmap='RdYlBu_r')
        plt.colorbar(im, label='Normalized Feature Value')
        plt.xlabel('Cluster')
        plt.ylabel('Feature Index')
        plt.title('Average Feature Values per Cluster (Normalized)')
        plt.xticks(range(n_clusters), [f'Cluster {i}' for i in unique_labels])
        
        # Create subplot for cluster sizes
        plt.subplot(2, 1, 2)
        cluster_sizes = [cluster_analysis[cluster_id]['size'] for cluster_id in unique_labels]
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        bars = plt.bar([f'Cluster {i}' for i in unique_labels], cluster_sizes, color=colors)
        
        plt.xlabel('Cluster')
        plt.ylabel('Number of Images')
        plt.title('Cluster Sizes')
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "cluster_comparison.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, feature_matrix, optimal_results, 
                                     clustering_results, cluster_analysis, images_info):
        """Generate comprehensive cluster analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")
        
        n_images = len(images_info)
        n_clusters = len(cluster_analysis)
        
        report = f"""
COMPREHENSIVE CLUSTER ANALYSIS REPORT
======================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

1. DATASET OVERVIEW
------------------
• Number of images analyzed: {n_images}
• Feature dimension: {feature_matrix.shape[1]}
• Total features extracted: {feature_matrix.size:,}

2. OPTIMAL CLUSTER DETERMINATION
-------------------------------
• Recommended number of clusters: {optimal_results['recommended']}
• By silhouette score: {optimal_results['by_silhouette']}
• By elbow method: {optimal_results['by_elbow']}
• By BIC: {optimal_results['by_bic']}
• By AIC: {optimal_results['by_aic']}

3. CLUSTERING RESULTS
-------------------
• Number of clusters used: {n_clusters}
• Clustering algorithms tested: {', '.join(clustering_results.keys())}

4. CLUSTER CHARACTERISTICS
------------------------
"""
        
        for cluster_id, analysis in cluster_analysis.items():
            report += f"""
Cluster {cluster_id}:
• Size: {analysis['size']} images ({analysis['size']/n_images*100:.1f}%)
• Characteristics: {', '.join(analysis['characteristics'])}
• Sample images: {', '.join([str(i) for i in analysis['indices'][:5]])}
"""
            if len(analysis['indices']) > 5:
                report += f"  ... and {len(analysis['indices']) - 5} more\n"
        
        report += f"""
5. CLUSTERING METRICS
-------------------
"""
        
        for method, result in clustering_results.items():
            if result is not None and 'silhouette' in result:
                report += f"""
{method.upper()}:
• Silhouette score: {result['silhouette']:.3f}
• Calinski-Harabasz: {result['calinski']:.1f}
• Davies-Bouldin: {result['davies']:.3f}
"""
        
        report += f"""
6. VISUALIZATIONS GENERATED
--------------------------
• cluster_metrics_comparison.png - Metrics for different cluster counts
• clustering_comparison.png - Results from different algorithms
• kmeans_detailed_analysis.png - Detailed KMeans visualization
• cluster_comparison.png - Heatmap of cluster features
• dendrogram_example.png - Hierarchical clustering dendrogram
• cluster_*_collage.png - Image collages for each cluster

7. DATA FILES
------------
• All cluster images saved in: {self.clusters_dir}/
• Cluster assignments saved in JSON format
• Feature matrices and labels saved as numpy files

8. INTERPRETATION GUIDELINES
--------------------------
1. Check if clusters correspond to visual categories (gender, age, ethnicity)
2. Look at cluster collages to identify common characteristics
3. Compare silhouette scores between methods
4. Examine which images are outliers or don't fit clusters well
5. Consider merging small clusters if they're not meaningful

9. POTENTIAL CATEGORIES IDENTIFIED
---------------------------------
Based on cluster characteristics, you might find:
• Gender-based clusters (male/female faces)
• Age-based clusters (young/old)
• Ethnicity-based clusters
• Lighting-based clusters (bright/dark images)
• Pose-based clusters (frontal/profile)
• Expression-based clusters (smiling/neutral)

10. NEXT STEPS
-------------
1. Manually inspect clusters to validate automatic categorization
2. Try different feature extraction methods
3. Experiment with different numbers of clusters
4. Use domain knowledge to label clusters meaningfully
5. Compare with human perception of similarity

========================================================================
Report generated by Detailed Cluster Analysis Framework
"""
        
        # Save report
        report_file = os.path.join(self.analysis_dir, "cluster_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f" Comprehensive report saved to: {report_file}")
        
        # Save cluster analysis data
        analysis_data = {
            'optimal_clusters': optimal_results,
            'clustering_results': {},
            'cluster_analysis': cluster_analysis,
            'image_info': images_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for method, result in clustering_results.items():
            if result is not None:
                analysis_data['clustering_results'][method] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        analysis_data['clustering_results'][method][key] = value.tolist()
                    else:
                        analysis_data['clustering_results'][method][key] = value
        
        with open(os.path.join(self.analysis_dir, "cluster_analysis_data.json"), 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f" Analysis data saved to: {os.path.join(self.analysis_dir, 'cluster_analysis_data.json')}")
        
        return report
    
    def run_analysis(self):
        """Run complete cluster analysis pipeline"""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE CLUSTER ANALYSIS")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Load images
        images_dir = "latent_analysis_20251202_175108/latents/"
        
        if not os.path.exists(images_dir):
            # Try alternative paths
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
                os.path.join(os.getcwd(), "latent_analysis_20251202_175108", "latents"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    images_dir = path
                    print(f"📂 Using images directory: {images_dir}")
                    break
        
        if not os.path.exists(images_dir):
            print(" No images directory found.")
            return None
        
        # Load images and features
        feature_matrix, images, image_info = self.load_images_and_features(images_dir)
        if feature_matrix is None:
            return None
        
        print(f"\n Starting analysis with {len(images)} images")
        
        # Step 1: Determine optimal number of clusters
        optimal_results, cluster_metrics = self.determine_optimal_clusters(feature_matrix)
        
        # Step 2: Perform clustering
        clustering_results, features_2d = self.perform_clustering(
            feature_matrix, optimal_results['recommended']
        )
        
        # Use KMeans labels for further analysis (as it's most commonly used)
        kmeans_labels = clustering_results['kmeans']['labels']
        
        # Step 3: Analyze cluster characteristics
        cluster_analysis = self.analyze_cluster_characteristics(
            images, image_info, kmeans_labels, feature_matrix
        )
        
        # Step 4: Generate comprehensive report
        report = self.generate_comprehensive_report(
            feature_matrix, optimal_results, clustering_results, cluster_analysis, image_info
        )
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        
        print(f"\n{'='*60}")
        print("CLUSTER ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        print(f"\n Summary:")
        print(f"   Duration: {elapsed_time}")
        print(f"   Images analyzed: {len(images)}")
        print(f"   Optimal clusters: {optimal_results['recommended']}")
        print(f"   Output directory: {self.output_dir}")
        
        print(f"\n Key Outputs:")
        print(f"   1. Cluster collages in: {self.clusters_dir}/")
        print(f"   2. Visualizations in: {self.visualizations_dir}/")
        print(f"   3. Analysis data in: {self.analysis_dir}/")
        
        print(f"\n🔍 What to Look For:")
        print(f"   1. Check if clusters show gender patterns")
        print(f"   2. Look for age groupings")
        print(f"   3. Identify ethnicity clusters")
        print(f"   4. Note any lighting or pose patterns")
        
        return {
            'feature_matrix': feature_matrix,
            'images': images,
            'image_info': image_info,
            'optimal_results': optimal_results,
            'clustering_results': clustering_results,
            'cluster_analysis': cluster_analysis,
            'output_dir': self.output_dir,
            'elapsed_time': elapsed_time
        }


def main():
    """Main function"""
    print("\n" + "="*60)
    print("DETAILED CLUSTER ANALYSIS FRAMEWORK")
    print("="*60)
    print("\nThis tool performs comprehensive cluster analysis to identify")
    print("natural groupings in your images (gender, age, ethnicity, etc.)\n")
    
    # Initialize analyzer
    analyzer = DetailedClusterAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print(f"\n Analysis completed successfully!")
        print(f" All outputs saved to: {results['output_dir']}")
        

    else:
        print("\n failed or was cancelled.")


if __name__ == "__main__":
    main()
