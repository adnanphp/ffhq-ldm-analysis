import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean
from scipy import interpolate
from scipy.stats import gaussian_kde
import warnings
from datetime import datetime
from tqdm import tqdm
import umap.umap_ as umap
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, KDTree
import warnings
warnings.filterwarnings('ignore')

class LatentManifoldAnalyzer:
    def __init__(self):
        """Initialize latent manifold analyzer"""
        print("🔬 LATENT MANIFOLD ANALYSIS")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"latent_manifold_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.interpolations_dir = os.path.join(self.output_dir, "interpolations")
        self.outliers_dir = os.path.join(self.output_dir, "outliers")
        
        for dir_path in [self.visualizations_dir, self.interpolations_dir, self.outliers_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_images_as_latents(self, images_dir):
        """Load images and extract features to use as pseudo-latents"""
        print(f"\n📂 Loading images from: {images_dir}")
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(png_files) == 0:
            print("❌ No PNG files found.")
            return None, None, None
        
        print(f"📊 Found {len(png_files)} images")
        
        # Load images and extract features
        images = []
        features = []
        image_info = []
        
        for idx, filename in enumerate(tqdm(png_files, desc="Processing images")):
            img_path = os.path.join(images_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img)
            images.append(img_np)
            
            # Extract features from image (simulating latent vectors)
            img_features = self.extract_image_features(img_np)
            features.append(img_features)
            
            image_info.append({
                'id': idx,
                'path': img_path,
                'filename': filename,
                'image': img
            })
        
        # Convert to numpy arrays
        images_array = np.array(images)
        features_array = np.array(features)
        
        print(f"✅ Extracted {features_array.shape[1]} features from images")
        print(f"✅ Features shape: {features_array.shape}")
        
        return features_array, images_array, image_info
    
    def extract_image_features(self, img):
        """Extract features from image to use as pseudo-latent vector"""
        # Convert to float and normalize
        img_float = img.astype(np.float32) / 255.0
        
        # Flatten the image
        flattened = img_float.flatten()
        
        # If image is too large, use PCA to reduce dimensionality
        if len(flattened) > 512:
            from sklearn.decomposition import PCA
            # Reshape to 2D for PCA
            n_samples = 1
            n_features = len(flattened)
            
            # For single image, we need to handle PCA differently
            # Instead, let's use a simpler approach
            # Downsample image
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((64, 64), PILImage.Resampling.LANCZOS)
            img_small = np.array(pil_img).astype(np.float32) / 255.0
            flattened = img_small.flatten()
        
        # Add some additional features
        features = []
        
        # Color statistics
        if len(img.shape) == 3:  # RGB image
            for channel in range(3):
                channel_data = img_float[:, :, channel]
                features.append(np.mean(channel_data))
                features.append(np.std(channel_data))
                features.append(np.percentile(channel_data, 25))
                features.append(np.percentile(channel_data, 75))
        
        # Texture features (simplified)
        from scipy.ndimage import sobel
        if len(img_float.shape) == 3:
            gray = np.mean(img_float, axis=2)
        else:
            gray = img_float
        
        # Edge features
        grad_x = sobel(gray)
        grad_y = sobel(gray, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(edge_magnitude))
        features.append(np.std(edge_magnitude))
        
        # Combine with flattened image
        all_features = np.concatenate([flattened[:100], np.array(features)])  # Limit to reasonable size
        
        return all_features
    
    def load_latents_and_images(self, data_dir):
        """Load data - tries to load latents, falls back to images"""
        print(f"\n📂 Looking for data in: {data_dir}")
        
        # First try to find latent files
        latent_extensions = ['.npy', '.npz', '.pt', '.pth']
        latent_files = []
        
        for ext in latent_extensions:
            files = [f for f in os.listdir(data_dir) if f.endswith(ext)]
            if files:
                latent_files = sorted(files)
                break
        
        if latent_files:
            print(f"📊 Found {len(latent_files)} latent files")
            return self.load_actual_latents(data_dir, latent_files)
        else:
            print("⚠️ No latent files found, using images as pseudo-latents")
            return self.load_images_as_latents(data_dir)
    
    def load_actual_latents(self, data_dir, latent_files):
        """Load actual latent vectors"""
        latents = []
        images = None
        latent_info = []
        
        for idx, filename in enumerate(tqdm(latent_files, desc="Loading latents")):
            latent_path = os.path.join(data_dir, filename)
            
            if filename.endswith('.npy'):
                latent = np.load(latent_path)
            elif filename.endswith('.npz'):
                data = np.load(latent_path)
                # Assuming first array contains latents
                latent = data[data.files[0]]
            else:
                print(f"⚠️ Skipping {filename} - format not supported")
                continue
            
            # Flatten if needed
            if latent.ndim > 1:
                latent = latent.flatten()
            
            latents.append(latent)
            latent_info.append({
                'id': idx,
                'path': latent_path,
                'filename': filename,
                'shape': latent.shape
            })
        
        latents = np.array(latents)
        
        # Try to load corresponding images
        images_dir = data_dir.replace('/latents', '/images') if '/latents' in data_dir else data_dir
        if os.path.exists(images_dir) and images_dir != data_dir:
            png_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            if png_files:
                images = []
                for filename in sorted(png_files)[:len(latents)]:
                    img_path = os.path.join(images_dir, filename)
                    img = Image.open(img_path)
                    img_np = np.array(img)
                    images.append(img_np)
                images = np.array(images)
        
        return latents, images, latent_info
    
    def perform_manifold_learning(self, latents, n_components=2):
        """Perform various manifold learning techniques"""
        print(f"\n🌐 Performing manifold learning (to {n_components}D)...")
        
        # Standardize data
        scaler = StandardScaler()
        latents_scaled = scaler.fit_transform(latents)
        
        manifold_results = {}
        
        # 1. PCA (Linear)
        print("   Performing PCA...")
        pca = PCA(n_components=n_components)
        manifold_results['pca'] = pca.fit_transform(latents_scaled)
        manifold_results['pca_variance'] = pca.explained_variance_ratio_
        
        # 2. t-SNE (Non-linear)
        print("   Performing t-SNE...")
        perplexity = min(30, len(latents) - 1)
        if perplexity < 1:
            perplexity = 5
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        manifold_results['tsne'] = tsne.fit_transform(latents_scaled)
        
        # 3. UMAP (Non-linear)
        print("   Performing UMAP...")
        try:
            umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
            manifold_results['umap'] = umap_reducer.fit_transform(latents_scaled)
        except:
            print("   ⚠️ UMAP failed, using PCA instead")
            manifold_results['umap'] = manifold_results['pca']
        
        # 4. Isomap (Non-linear, preserves geodesic distances)
        print("   Performing Isomap...")
        n_neighbors = min(10, len(latents) - 1)
        if n_neighbors > 2:
            isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            manifold_results['isomap'] = isomap.fit_transform(latents_scaled)
        else:
            print("   ⚠️ Not enough samples for Isomap, using PCA")
            manifold_results['isomap'] = manifold_results['pca']
        
        # 5. MDS (Multi-dimensional scaling)
        print("   Performing MDS...")
        if len(latents) > 10:
            mds = MDS(n_components=n_components, random_state=42, dissimilarity='euclidean')
            manifold_results['mds'] = mds.fit_transform(latents_scaled)
        else:
            print("   ⚠️ Not enough samples for MDS, using PCA")
            manifold_results['mds'] = manifold_results['pca']
        
        # Calculate manifold properties
        self.calculate_manifold_properties(latents_scaled, manifold_results)
        
        return manifold_results
    
    def calculate_manifold_properties(self, latents, manifold_results):
        """Calculate topological properties of manifolds"""
        print("\n📐 Calculating manifold properties...")
        
        properties = {}
        
        for method, embedding in manifold_results.items():
            if method == 'pca_variance':
                continue
            
            if len(embedding) < 5:
                print(f"   ⚠️ Not enough points for {method} properties")
                continue
            
            try:
                # 1. Local density variation
                tree = KDTree(embedding)
                k = min(5, len(embedding) - 1)
                distances, _ = tree.query(embedding, k=k+1)  # k nearest neighbors + self
                avg_local_density = np.mean(1 / (distances[:, 1:].mean(axis=1) + 1e-8))
                density_variation = np.std(1 / (distances[:, 1:].mean(axis=1) + 1e-8))
                
                # 2. Curvature estimation (simplified)
                curvatures = []
                for i in range(len(embedding)):
                    if i < 5 or i >= len(embedding) - 5:
                        continue
                    
                    # Local neighborhood
                    neighbors = embedding[max(0, i-3):min(len(embedding), i+3)]
                    if len(neighbors) < 3:
                        continue
                    
                    # Simple curvature estimation
                    center = embedding[i]
                    distances_to_center = np.linalg.norm(neighbors - center, axis=1)
                    avg_distance = np.mean(distances_to_center)
                    std_distance = np.std(distances_to_center)
                    curvature = std_distance / (avg_distance + 1e-8)
                    curvatures.append(curvature)
                
                avg_curvature = np.mean(curvatures) if curvatures else 0
                
                # 3. Connectivity - Minimum Spanning Tree
                from scipy.spatial.distance import pdist, squareform
                dist_matrix = squareform(pdist(embedding))
                mst = minimum_spanning_tree(dist_matrix)
                mst_density = mst.sum() / (len(embedding) - 1) if len(embedding) > 1 else 0
                
                # 4. Intrinsic dimensionality estimate
                if len(embedding) > 10:
                    nbrs = NearestNeighbors(n_neighbors=min(10, len(embedding)-1)).fit(embedding)
                    distances, _ = nbrs.kneighbors(embedding)
                    r1 = distances[:, 1]  # Distance to 1st neighbor
                    r2 = distances[:, 2]  # Distance to 2nd neighbor
                    dim_estimate = np.mean(np.log(r2 / r1) / np.log(2))
                else:
                    dim_estimate = embedding.shape[1]
                
                properties[method] = {
                    'avg_local_density': float(avg_local_density),
                    'density_variation': float(density_variation),
                    'avg_curvature': float(avg_curvature),
                    'mst_density': float(mst_density),
                    'dim_estimate': float(dim_estimate),
                    'n_points': len(embedding)
                }
                
                print(f"   {method.upper():8s}: Density={avg_local_density:.3f}, "
                      f"Curvature={avg_curvature:.3f}, Dim≈{dim_estimate:.1f}")
                
            except Exception as e:
                print(f"   ⚠️ Could not calculate properties for {method}: {e}")
                properties[method] = {
                    'avg_local_density': 0,
                    'density_variation': 0,
                    'avg_curvature': 0,
                    'mst_density': 0,
                    'dim_estimate': embedding.shape[1],
                    'n_points': len(embedding)
                }
        
        self.manifold_properties = properties
        return properties
    
    def visualize_manifolds(self, manifold_results, images=None, latent_info=None):
        """Create visualizations of different manifold embeddings"""
        print("\n🎨 Creating manifold visualizations...")
        
        methods = ['pca', 'tsne', 'umap', 'isomap', 'mds']
        
        # Create subplot grid
        n_methods = len([m for m in methods if m in manifold_results])
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        if n_methods == 0:
            print("⚠️ No manifold results to visualize")
            return
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        plot_idx = 0
        for method in methods:
            if method not in manifold_results:
                continue
            
            embedding = manifold_results[method]
            
            plt.subplot(n_rows, n_cols, plot_idx + 1)
            plot_idx += 1
            
            # Color by density
            try:
                tree = KDTree(embedding)
                k = min(5, len(embedding) - 1)
                distances, _ = tree.query(embedding, k=k+1)
                densities = 1 / (distances[:, 1:].mean(axis=1) + 1e-8)
                
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=densities, cmap='viridis', 
                                    alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
                plt.colorbar(scatter, label='Local Density')
            except:
                plt.scatter(embedding[:, 0], embedding[:, 1], 
                          alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
            
            # Add convex hull if enough points
            if len(embedding) > 3:
                try:
                    hull = ConvexHull(embedding)
                    for simplex in hull.simplices:
                        plt.plot(embedding[simplex, 0], embedding[simplex, 1], 
                                'r--', alpha=0.5, linewidth=1)
                except:
                    pass
            
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title(f'{method.upper()} Embedding')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('Latent Manifold Learning Visualizations', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "manifold_embeddings.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def perform_interpolation_analysis(self, latents, n_steps=10):
        """Analyze interpolation quality in latent space"""
        print(f"\n🔗 Performing interpolation analysis ({n_steps} steps)...")
        
        if len(latents) < 2:
            print("⚠️ Need at least 2 points for interpolation analysis")
            return {}
        
        interpolation_results = {}
        
        # Select diverse pairs of points for interpolation
        n_pairs = min(3, len(latents) // 2)
        pairs = self.select_diverse_pairs(latents, n_pairs)
        
        for pair_idx, (i, j) in enumerate(pairs):
            print(f"   Pair {pair_idx + 1}: Points {i} ↔ {j}")
            
            # Extract latent vectors
            z1 = latents[i]
            z2 = latents[j]
            
            # 1. Linear interpolation
            linear_interp = []
            for alpha in np.linspace(0, 1, n_steps):
                z_interp = (1 - alpha) * z1 + alpha * z2
                linear_interp.append(z_interp)
            linear_interp = np.array(linear_interp)
            
            # 2. Spherical interpolation (SLERP)
            spherical_interp = []
            for alpha in np.linspace(0, 1, n_steps):
                # Normalize to unit sphere
                norm1 = np.linalg.norm(z1)
                norm2 = np.linalg.norm(z2)
                
                if norm1 < 1e-8 or norm2 < 1e-8:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                else:
                    z1_norm = z1 / norm1
                    z2_norm = z2 / norm2
                    
                    # Calculate angle between vectors
                    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0, 1.0)
                    omega = np.arccos(dot)
                    
                    if np.abs(omega) < 1e-8:
                        z_interp = z1
                    else:
                        z_interp = (np.sin((1 - alpha) * omega) * z1_norm + 
                                   np.sin(alpha * omega) * z2_norm) / np.sin(omega)
                
                spherical_interp.append(z_interp)
            spherical_interp = np.array(spherical_interp)
            
            # Calculate interpolation quality metrics
            linear_metrics = self.calculate_interpolation_metrics(linear_interp, z1, z2)
            spherical_metrics = self.calculate_interpolation_metrics(spherical_interp, z1, z2)
            
            interpolation_results[pair_idx] = {
                'point_indices': [int(i), int(j)],
                'linear_interpolation': linear_interp.tolist(),
                'spherical_interpolation': spherical_interp.tolist(),
                'linear_metrics': linear_metrics,
                'spherical_metrics': spherical_metrics,
                'comparison': {
                    'smoothness_ratio': spherical_metrics['smoothness'] / (linear_metrics['smoothness'] + 1e-8),
                    'curvature_ratio': spherical_metrics['avg_curvature'] / (linear_metrics['avg_curvature'] + 1e-8),
                    'path_length_ratio': spherical_metrics['path_length'] / (linear_metrics['path_length'] + 1e-8)
                }
            }
            
            print(f"      Linear: Smoothness={linear_metrics['smoothness']:.3f}, "
                  f"Curvature={linear_metrics['avg_curvature']:.3f}")
            print(f"      Spherical: Smoothness={spherical_metrics['smoothness']:.3f}, "
                  f"Curvature={spherical_metrics['avg_curvature']:.3f}")
        
        # Create interpolation visualizations
        if interpolation_results:
            self.visualize_interpolations(interpolation_results, latents)
        
        return interpolation_results
    
    def select_diverse_pairs(self, latents, n_pairs):
        """Select diverse pairs of points for interpolation"""
        if len(latents) < 2:
            return []
        
        # Simple approach: use first and last, middle points
        pairs = []
        
        if len(latents) >= 2:
            # Pair 1: first and last
            pairs.append((0, len(latents)-1))
        
        if len(latents) >= 4 and n_pairs > 1:
            # Pair 2: first quarter and third quarter
            idx1 = len(latents) // 4
            idx2 = 3 * len(latents) // 4
            pairs.append((idx1, idx2))
        
        if len(latents) >= 6 and n_pairs > 2:
            # Pair 3: middle points
            idx1 = len(latents) // 3
            idx2 = 2 * len(latents) // 3
            pairs.append((idx1, idx2))
        
        return pairs[:n_pairs]
    
    def calculate_interpolation_metrics(self, interp_points, start, end):
        """Calculate quality metrics for interpolation"""
        metrics = {}
        
        # 1. Path length
        path_length = 0
        for k in range(len(interp_points) - 1):
            path_length += np.linalg.norm(interp_points[k+1] - interp_points[k])
        metrics['path_length'] = float(path_length)
        
        # 2. Smoothness (second derivative)
        second_derivatives = []
        for k in range(1, len(interp_points) - 1):
            d2 = interp_points[k+1] - 2*interp_points[k] + interp_points[k-1]
            second_derivatives.append(np.linalg.norm(d2))
        
        if second_derivatives:
            metrics['smoothness'] = float(1 / (np.mean(second_derivatives) + 1e-8))
        else:
            metrics['smoothness'] = 0
        
        # 3. Curvature
        curvatures = []
        for k in range(1, len(interp_points) - 1):
            v1 = interp_points[k] - interp_points[k-1]
            v2 = interp_points[k+1] - interp_points[k]
            if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                curvature = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
                curvatures.append(curvature)
        metrics['avg_curvature'] = float(np.mean(curvatures)) if curvatures else 0
        
        # 4. Distance from endpoints
        metrics['start_distance'] = float(np.linalg.norm(interp_points[0] - start))
        metrics['end_distance'] = float(np.linalg.norm(interp_points[-1] - end))
        
        # 5. Linearity (deviation from straight line)
        deviation = 0
        for k, point in enumerate(interp_points):
            alpha = k / (len(interp_points) - 1) if len(interp_points) > 1 else 0
            straight_point = (1 - alpha) * start + alpha * end
            deviation += np.linalg.norm(point - straight_point)
        
        if len(interp_points) > 0:
            metrics['linearity_deviation'] = float(deviation / len(interp_points))
        else:
            metrics['linearity_deviation'] = 0
        
        return metrics
    
    def visualize_interpolations(self, interpolation_results, latents):
        """Visualize interpolation results"""
        print("\n📈 Creating interpolation visualizations...")
        
        # Project latents to 2D for visualization
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        n_pairs = len(interpolation_results)
        fig, axes = plt.subplots(2, n_pairs, figsize=(5*n_pairs, 8))
        
        if n_pairs == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        for pair_idx, results in interpolation_results.items():
            i, j = results['point_indices']
            
            # Plot in latent space
            ax_top = axes[0, pair_idx]
            ax_bottom = axes[1, pair_idx]
            
            # Plot all points
            ax_top.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.3, s=10, c='gray')
            ax_bottom.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.3, s=10, c='gray')
            
            # Highlight interpolation points
            ax_top.scatter([latents_2d[i, 0], latents_2d[j, 0]], 
                         [latents_2d[i, 1], latents_2d[j, 1]], 
                         c=['red', 'blue'], s=100, edgecolors='black')
            ax_bottom.scatter([latents_2d[i, 0], latents_2d[j, 0]], 
                            [latents_2d[i, 1], latents_2d[j, 1]], 
                            c=['red', 'blue'], s=100, edgecolors='black')
            
            # Convert interpolations to 2D
            linear_interp_2d = pca.transform(np.array(results['linear_interpolation']))
            spherical_interp_2d = pca.transform(np.array(results['spherical_interpolation']))
            
            # Plot linear interpolation
            ax_top.plot(linear_interp_2d[:, 0], linear_interp_2d[:, 1], 
                       'g-', linewidth=2, alpha=0.7, label='Linear')
            ax_top.scatter(linear_interp_2d[:, 0], linear_interp_2d[:, 1], 
                          c='green', s=30, alpha=0.5)
            
            # Plot spherical interpolation
            ax_bottom.plot(spherical_interp_2d[:, 0], spherical_interp_2d[:, 1], 
                          'purple', linewidth=2, alpha=0.7, label='Spherical')
            ax_bottom.scatter(spherical_interp_2d[:, 0], spherical_interp_2d[:, 1], 
                            c='purple', s=30, alpha=0.5)
            
            # Add metrics text
            linear_metrics = results['linear_metrics']
            spherical_metrics = results['spherical_metrics']
            
            ax_top.text(0.05, 0.95, 
                       f"Length: {linear_metrics['path_length']:.2f}\n"
                       f"Smoothness: {linear_metrics['smoothness']:.2f}",
                       transform=ax_top.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax_bottom.text(0.05, 0.95, 
                          f"Length: {spherical_metrics['path_length']:.2f}\n"
                          f"Smoothness: {spherical_metrics['smoothness']:.2f}",
                          transform=ax_bottom.transAxes, fontsize=8,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax_top.set_title(f'Pair {pair_idx+1}: Linear Interpolation')
            ax_bottom.set_title(f'Pair {pair_idx+1}: Spherical Interpolation')
            ax_top.legend()
            ax_bottom.legend()
            ax_top.grid(True, alpha=0.3)
            ax_bottom.grid(True, alpha=0.3)
        
        plt.suptitle('Interpolation Analysis: Linear vs Spherical', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "interpolation_comparison.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def perform_density_estimation(self, latents):
        """Estimate probability density in latent space"""
        print("\n📊 Performing density estimation...")
        
        if len(latents) < 5:
            print("⚠️ Need at least 5 points for density estimation")
            return {}
        
        density_results = {}
        
        # 1. Kernel Density Estimation (KDE)
        print("   Performing Kernel Density Estimation...")
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(latents)
            log_density = kde.score_samples(latents)
            density = np.exp(log_density)
            
            density_results['kde'] = {
                'density': density.tolist(),
                'log_density': log_density.tolist(),
                'bandwidth': kde.bandwidth,
                'mean_density': float(np.mean(density)),
                'std_density': float(np.std(density))
            }
        except Exception as e:
            print(f"   ⚠️ KDE failed: {e}")
        
        # 2. Gaussian Mixture Model
        print("   Fitting Gaussian Mixture Model...")
        try:
            n_components = min(5, len(latents) // 3)
            if n_components >= 1:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(latents)
                gmm_density = np.exp(gmm.score_samples(latents))
                
                density_results['gmm'] = {
                    'density': gmm_density.tolist(),
                    'n_components': gmm.n_components,
                    'mean_density': float(np.mean(gmm_density)),
                    'std_density': float(np.std(gmm_density))
                }
        except Exception as e:
            print(f"   ⚠️ GMM failed: {e}")
        
        # 3. k-NN density estimation
        print("   Performing k-NN density estimation...")
        try:
            k = min(10, len(latents) - 1)
            knn_density = self.knn_density_estimation(latents, k=k)
            
            density_results['knn'] = {
                'density': knn_density.tolist(),
                'mean_density': float(np.mean(knn_density)),
                'std_density': float(np.std(knn_density))
            }
        except Exception as e:
            print(f"   ⚠️ k-NN density estimation failed: {e}")
        
        # Create density visualizations
        if density_results:
            self.visualize_density_estimation(latents, density_results)
        
        return density_results
    
    def knn_density_estimation(self, data, k=10):
        """k-NN density estimation"""
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(data))).fit(data)
        distances, _ = nbrs.kneighbors(data)
        
        # Volume of k-nearest neighbors ball
        volumes = distances[:, k] ** data.shape[1]
        
        # Density estimate: k / (n * volume)
        density = k / (len(data) * volumes + 1e-8)
        
        return density
    
    def visualize_density_estimation(self, latents, density_results):
        """Visualize density estimation results"""
        print("\n📈 Creating density visualizations...")
        
        # Project to 2D for visualization
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        # Create subplots
        n_methods = len(density_results)
        if n_methods == 0:
            return
        
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = np.array([axes])
        
        plot_idx = 0
        for method, results in density_results.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if 'density' in results:
                density = np.array(results['density'])
                
                # Plot points colored by density
                scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                                   c=density, cmap='viridis', s=30, alpha=0.8)
                
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f'{method.upper()} Density')
                ax.grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=ax, label='Density')
            
            plot_idx += 1
        
        plt.suptitle('Latent Space Density Estimation', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "density_estimation.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def perform_outlier_detection(self, latents, images=None, latent_info=None):
        """Detect outliers in latent space"""
        print("\n🔍 Performing outlier detection...")
        
        if len(latents) < 10:
            print("⚠️ Need at least 10 points for reliable outlier detection")
            return {}
        
        outlier_results = {}
        
        # 1. Isolation Forest
        print("   Running Isolation Forest...")
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_predictions = iso_forest.fit_predict(latents)
            iso_scores = iso_forest.decision_function(latents)
            
            outlier_results['isolation_forest'] = {
                'predictions': iso_predictions.tolist(),
                'scores': iso_scores.tolist(),
                'outlier_indices': np.where(iso_predictions == -1)[0].tolist()
            }
        except Exception as e:
            print(f"   ⚠️ Isolation Forest failed: {e}")
        
        # 2. Local Outlier Factor
        print("   Running Local Outlier Factor...")
        try:
            n_neighbors = min(20, len(latents) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
            lof_predictions = lof.fit_predict(latents)
            lof_scores = lof.negative_outlier_factor_
            
            outlier_results['local_outlier_factor'] = {
                'predictions': lof_predictions.tolist(),
                'scores': lof_scores.tolist(),
                'outlier_indices': np.where(lof_predictions == -1)[0].tolist()
            }
        except Exception as e:
            print(f"   ⚠️ Local Outlier Factor failed: {e}")
        
        # 3. One-Class SVM
        print("   Running One-Class SVM...")
        try:
            oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            oc_svm_predictions = oc_svm.fit_predict(latents)
            oc_svm_scores = oc_svm.decision_function(latents)
            
            outlier_results['one_class_svm'] = {
                'predictions': oc_svm_predictions.tolist(),
                'scores': oc_svm_scores.tolist(),
                'outlier_indices': np.where(oc_svm_predictions == -1)[0].tolist()
            }
        except Exception as e:
            print(f"   ⚠️ One-Class SVM failed: {e}")
        
        # Find consensus outliers
        if outlier_results:
            consensus_outliers = self.find_consensus_outliers(outlier_results)
            outlier_results['consensus'] = {
                'outlier_indices': consensus_outliers.tolist(),
                'n_outliers': len(consensus_outliers)
            }
            
            print(f"✅ Found {len(consensus_outliers)} consensus outliers")
        
        # Create outlier visualizations
        if outlier_results:
            self.visualize_outliers(latents, outlier_results, images, latent_info)
        
        return outlier_results
    
    def find_consensus_outliers(self, outlier_results):
        """Find outliers detected by multiple methods"""
        n_points = len(outlier_results[list(outlier_results.keys())[0]]['predictions'])
        outlier_votes = np.zeros(n_points)
        
        # Methods to consider for consensus
        methods = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
        
        for method in methods:
            if method in outlier_results:
                outlier_indices = outlier_results[method]['outlier_indices']
                outlier_votes[outlier_indices] += 1
        
        # Points flagged by at least 2 methods
        consensus_threshold = 2
        consensus_outliers = np.where(outlier_votes >= consensus_threshold)[0]
        
        return consensus_outliers
    
    def visualize_outliers(self, latents, outlier_results, images=None, latent_info=None):
        """Visualize outlier detection results"""
        print("\n📊 Creating outlier visualizations...")
        
        # Project to 2D for visualization
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        methods = ['isolation_forest', 'local_outlier_factor', 'one_class_svm', 'consensus']
        method_names = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'Consensus']
        
        for idx, (method, name) in enumerate(zip(methods, method_names)):
            if idx >= len(axes) or method not in outlier_results:
                continue
            
            ax = axes[idx]
            
            # Get outlier indices
            outlier_indices = outlier_results[method]['outlier_indices']
            normal_indices = [i for i in range(len(latents)) if i not in outlier_indices]
            
            # Plot normal points
            if normal_indices:
                ax.scatter(latents_2d[normal_indices, 0], latents_2d[normal_indices, 1],
                          c='blue', s=20, alpha=0.5, label='Normal')
            
            # Plot outliers
            if outlier_indices:
                ax.scatter(latents_2d[outlier_indices, 0], latents_2d[outlier_indices, 1],
                          c='red', s=100, edgecolors='black', linewidth=1.5,
                          label=f'Outliers ({len(outlier_indices)})')
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'{name} Outlier Detection')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Outlier Detection in Latent Space', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "outlier_detection.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save outlier images if available
        if images is not None and 'consensus' in outlier_results:
            self.save_outlier_images(outlier_results, images, latent_info)
    
    def save_outlier_images(self, outlier_results, images, latent_info):
        """Save images of detected outliers"""
        if 'consensus' not in outlier_results:
            return
        
        consensus_outliers = outlier_results['consensus']['outlier_indices']
        
        print(f"\n💾 Saving {len(consensus_outliers)} outlier images...")
        
        # Create directory for outlier images
        outlier_images_dir = os.path.join(self.outliers_dir, "images")
        os.makedirs(outlier_images_dir, exist_ok=True)
        
        # Save individual outlier images
        for idx, outlier_idx in enumerate(consensus_outliers):
            if outlier_idx < len(images):
                img = images[outlier_idx]
                
                # Convert numpy array to PIL Image
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = img
                
                # Save image
                filename = f"outlier_{idx:03d}_idx{outlier_idx}.png"
                filepath = os.path.join(outlier_images_dir, filename)
                pil_img.save(filepath)
        
        # Create outlier collage
        self.create_outlier_collage(images, consensus_outliers, outlier_images_dir)
    
    def create_outlier_collage(self, images, outlier_indices, output_dir):
        """Create collage of outlier images"""
        if len(outlier_indices) == 0:
            return
        
        # Select up to 16 outliers for collage
        n_outliers = min(16, len(outlier_indices))
        selected_indices = outlier_indices[:n_outliers]
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(n_outliers)))
        thumb_size = 120
        
        collage_width = grid_size * thumb_size
        collage_height = grid_size * thumb_size
        collage = Image.new('RGB', (collage_width, collage_height), color='white')
        
        for idx, img_idx in enumerate(selected_indices):
            if img_idx >= len(images):
                continue
            
            row = idx // grid_size
            col = idx % grid_size
            
            img_array = images[img_idx]
            if isinstance(img_array, np.ndarray):
                img = Image.fromarray(img_array)
            else:
                img = img_array
            
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            x = col * thumb_size
            y = row * thumb_size
            
            collage.paste(img, (x, y))
            
            # Add outlier number
            from PIL import ImageDraw
            draw = ImageDraw.Draw(collage)
            draw.text((x + 5, y + 5), str(idx), 
                     fill='white', stroke_width=2, stroke_fill='red')
        
        # Save collage
        collage.save(os.path.join(output_dir, "outlier_collage.png"))
        print(f"✅ Created outlier collage with {n_outliers} images")
    
    def generate_report(self, latents, manifold_results, interpolation_results, 
                       density_results, outlier_results):
        """Generate analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING ANALYSIS REPORT")
        print(f"{'='*60}")
        
        n_points = len(latents)
        latent_dim = latents.shape[1] if latents.ndim > 1 else 1
        
        report = f"""
LATENT SPACE ANALYSIS REPORT
============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

1. DATASET SUMMARY
-----------------
• Total points analyzed: {n_points}
• Feature dimensionality: {latent_dim}
• Data range: [{latents.min():.3f}, {latents.max():.3f}]
• Data mean: {latents.mean():.3f}, std: {latents.std():.3f}

2. MANIFOLD PROPERTIES
---------------------
"""
        
        if hasattr(self, 'manifold_properties'):
            for method, props in self.manifold_properties.items():
                report += f"""
{method.upper()}:
• Intrinsic dimension estimate: {props['dim_estimate']:.2f}
• Average local density: {props['avg_local_density']:.3f}
• Average curvature: {props['avg_curvature']:.3f}
"""
        
        report += f"""
3. DENSITY ESTIMATION
--------------------
"""
        
        if density_results:
            for method, results in density_results.items():
                report += f"""
{method.upper()}:
• Average density: {results.get('mean_density', 0):.3e}
• Density std: {results.get('std_density', 0):.3e}
"""
        
        report += f"""
4. OUTLIER DETECTION
-------------------
"""
        
        if outlier_results and 'consensus' in outlier_results:
            consensus_outliers = outlier_results['consensus']['outlier_indices']
            n_outliers = len(consensus_outliers)
            outlier_percentage = (n_outliers / n_points) * 100 if n_points > 0 else 0
            
            report += f"""
• Total consensus outliers: {n_outliers} ({outlier_percentage:.1f}%)
• Outlier indices: {consensus_outliers[:10]}{'...' if len(consensus_outliers) > 10 else ''}
"""
        
        report += f"""
5. INTERPOLATION ANALYSIS
------------------------
"""
        
        if interpolation_results:
            for pair_idx, results in interpolation_results.items():
                report += f"""
Pair {pair_idx + 1}:
• Linear smoothness: {results['linear_metrics']['smoothness']:.3f}
• Spherical smoothness: {results['spherical_metrics']['smoothness']:.3f}
• Improvement: {((results['spherical_metrics']['smoothness'] - results['linear_metrics']['smoothness']) / results['linear_metrics']['smoothness'] * 100):+.1f}%
"""
        
        report += f"""
6. VISUALIZATIONS
---------------
• Manifold embeddings: {self.visualizations_dir}/manifold_embeddings.png
• Density estimation: {self.visualizations_dir}/density_estimation.png
• Outlier detection: {self.visualizations_dir}/outlier_detection.png
• Interpolation analysis: {self.visualizations_dir}/interpolation_comparison.png

========================================================================
"""
        
        # Save report
        report_file = os.path.join(self.output_dir, "analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_file}")
        
        return report
    
    def run_analysis(self, data_path=None):
        """Run comprehensive latent space analysis"""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE LATENT SPACE ANALYSIS")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Determine data path
        if data_path is None:
            # Try common paths
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
                "../latent_analysis_20251202_175108/latents/",
                "latents/",
                "images/"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"📂 Using data directory: {data_path}")
                    break
        
        if data_path is None or not os.path.exists(data_path):
            print("❌ No data directory found.")
            print("💡 Please specify the path to your data directory.")
            return None
        
        # Load data
        latents, images, latent_info = self.load_latents_and_images(data_path)
        if latents is None:
            return None
        
        print(f"\n✅ Starting analysis with {len(latents)} data points")
        
        # 1. Manifold Learning
        manifold_results = self.perform_manifold_learning(latents, n_components=2)
        self.visualize_manifolds(manifold_results, images, latent_info)
        
        # 2. Interpolation Analysis
        interpolation_results = self.perform_interpolation_analysis(latents, n_steps=10)
        
        # 3. Density Estimation
        density_results = self.perform_density_estimation(latents)
        
        # 4. Outlier Detection
        outlier_results = self.perform_outlier_detection(latents, images, latent_info)
        
        # 5. Generate Report
        report = self.generate_report(latents, manifold_results, interpolation_results,
                                     density_results, outlier_results)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Total time: {elapsed_time.total_seconds():.1f} seconds")
        print(f"✅ Output saved to: {self.output_dir}/")
        print(f"✅ Report: {self.output_dir}/analysis_report.txt")
        print(f"\n🎯 Analysis Summary:")
        print(f"   • Analyzed {len(latents)} data points")
        print(f"   • Studied manifold topology")
        print(f"   • Compared interpolation methods")
        print(f"   • Estimated probability density")
        print(f"   • Identified anomalous points")
        
        return {
            'latents': latents,
            'images': images,
            'latent_info': latent_info,
            'manifold_results': manifold_results,
            'interpolation_results': interpolation_results,
            'density_results': density_results,
            'outlier_results': outlier_results,
            'output_dir': self.output_dir
        }


# Main execution
if __name__ == "__main__":
    analyzer = LatentManifoldAnalyzer()
    
    # You can specify the path if needed:
    # results = analyzer.run_analysis("path/to/your/images")
    
    # Or let it auto-detect:
    results = analyzer.run_analysis()
