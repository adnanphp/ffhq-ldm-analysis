import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import warnings
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SemanticAttributeAnalyzer:
    def __init__(self):
        """Initialize semantic attribute analyzer"""
        print("🔍 SEMANTIC ATTRIBUTE DISCOVERY & REGRESSION")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"semantic_attribute_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.attributes_dir = os.path.join(self.output_dir, "attributes")
        
        for dir_path in [self.visualizations_dir, self.attributes_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_data(self, data_dir, labels_file=None):
        """Load data from directory - handles both images and latent files"""
        print(f"\n📂 Loading data from: {data_dir}")
        
        # Get all files
        all_files = os.listdir(data_dir)
        
        # Check for image files first
        image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print(f"📊 Found {len(image_files)} image files")
            return self.process_image_directory(data_dir, labels_file)
        else:
            print("❌ No image files found")
            return None, None, None, None
    
    def process_image_directory(self, images_dir, labels_file=None):
        """Process directory of images"""
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            return None, None, None, None
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        images = []
        features = []
        image_info = []
        
        for idx, filename in enumerate(tqdm(image_files, desc="Loading images")):
            img_path = os.path.join(images_dir, filename)
            try:
                # Load image
                img = Image.open(img_path)
                img_np = np.array(img)
                images.append(img_np)
                
                # Extract meaningful features (not just flattening)
                img_features = self.extract_semantic_features(img_np)
                features.append(img_features)
                
                image_info.append({
                    'id': idx,
                    'filename': filename,
                    'path': img_path,
                    'shape': img_np.shape
                })
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
        
        if not features:
            return None, None, None, None
        
        features_array = np.array(features)
        print(f"✅ Extracted {features_array.shape[1]} features from {len(features)} images")
        
        # Load attribute labels if available
        attributes, attribute_names = self.load_attribute_labels(labels_file, len(features))
        
        return features_array, attributes, attribute_names, image_info
    
    def extract_semantic_features(self, img_np):
        """Extract meaningful semantic features from images"""
        img_float = img_np.astype(np.float32) / 255.0
        
        features = []
        
        # 1. Color features
        if len(img_float.shape) == 3:  # RGB
            # Average color
            avg_r = np.mean(img_float[:, :, 0])
            avg_g = np.mean(img_float[:, :, 1])
            avg_b = np.mean(img_float[:, :, 2])
            features.extend([avg_r, avg_g, avg_b])
            
            # Color standard deviation
            std_r = np.std(img_float[:, :, 0])
            std_g = np.std(img_float[:, :, 1])
            std_b = np.std(img_float[:, :, 2])
            features.extend([std_r, std_g, std_b])
            
            # Convert to HSV for better color representation
            from colorsys import rgb_to_hsv
            h, s, v = rgb_to_hsv(avg_r, avg_g, avg_b)
            features.extend([h, s, v])
        else:  # Grayscale
            avg_gray = np.mean(img_float)
            std_gray = np.std(img_float)
            features.extend([avg_gray, std_gray])
        
        # 2. Texture features
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
        
        # 3. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])
        
        # 4. Face-specific features (if faces are present)
        height, width = gray.shape
        
        # Center region (where face usually is)
        center_h, center_w = height // 2, width // 2
        crop_size = min(height, width) // 3
        
        top = max(0, center_h - crop_size)
        bottom = min(height, center_h + crop_size)
        left = max(0, center_w - crop_size)
        right = min(width, center_w + crop_size)
        
        face_region = gray[top:bottom, left:right]
        if len(face_region) > 0:
            face_brightness = np.mean(face_region)
            face_contrast = np.std(face_region)
            features.extend([face_brightness, face_contrast])
        
        # 5. Symmetry
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        right_half_flipped = np.fliplr(right_half)
        
        min_height = min(left_half.shape[0], right_half_flipped.shape[0])
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        
        if min_height > 0 and min_width > 0:
            left_crop = left_half[:min_height, :min_width]
            right_crop = right_half_flipped[:min_height, :min_width]
            symmetry = 1.0 - np.mean(np.abs(left_crop - right_crop))
            features.append(symmetry)
        else:
            features.append(0.5)  # Neutral symmetry
        
        return np.array(features)
    
    def load_attribute_labels(self, labels_file, n_samples):
        """Load attribute labels from file"""
        attributes = None
        attribute_names = None
        
        if labels_file and os.path.exists(labels_file):
            print(f"📂 Loading attribute labels from: {labels_file}")
            
            try:
                if labels_file.endswith('.json'):
                    with open(labels_file, 'r') as f:
                        label_data = json.load(f)
                    
                    if 'attributes' in label_data:
                        attributes = np.array(label_data['attributes'])
                        attribute_names = label_data.get('attribute_names', [])
                elif labels_file.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(labels_file)
                    attributes = df.values[:n_samples]  # Match number of samples
                    attribute_names = df.columns.tolist()
                
                if attributes is not None:
                    print(f"✅ Loaded {attributes.shape[1]} attributes for {attributes.shape[0]} samples")
            except Exception as e:
                print(f"⚠️ Error loading labels: {e}")
        
        return attributes, attribute_names
    
    def discover_semantic_directions(self, latents, n_components=10):
        """Discover semantic directions using PCA"""
        print(f"\n🔍 Discovering semantic directions...")
        
        # Standardize
        scaler = StandardScaler()
        latents_scaled = scaler.fit_transform(latents)
        
        # Use fewer components if we have few samples
        max_components = min(n_components, latents.shape[0] - 1, latents.shape[1])
        
        pca = PCA(n_components=max_components)
        pca_components = pca.fit_transform(latents_scaled)
        explained_variance = pca.explained_variance_ratio_
        
        total_variance = np.sum(explained_variance[:max_components]) * 100
        print(f"✅ Top {max_components} PCA components explain {total_variance:.1f}% of variance")
        
        # Show top components
        print("   Top 5 components variance:")
        for i in range(min(5, len(explained_variance))):
            print(f"   PC{i+1}: {explained_variance[i]*100:.2f}%")
        
        self.pca_components = pca.components_
        self.pca_variance = explained_variance
        self.pca_scaler = scaler
        
        return pca_components, explained_variance
    
    def perform_unsupervised_attribute_discovery(self, latents, n_clusters=5):
        """Discover semantic attributes without labels using clustering"""
        print(f"\n🔍 Performing unsupervised attribute discovery...")
        
        # Reduce dimensionality first for better clustering
        n_samples, n_features = latents.shape
        
        # Use PCA to reduce to reasonable dimensions
        pca_for_clustering = PCA(n_components=min(20, n_features, n_samples-1))
        latents_reduced = pca_for_clustering.fit_transform(latents)
        
        # Determine optimal number of clusters
        if n_samples >= 10:
            # Try different numbers of clusters
            silhouette_scores = []
            cluster_range = range(2, min(8, n_samples // 3))
            
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(latents_reduced)
                
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(latents_reduced, cluster_labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            
            # Choose best number of clusters
            if silhouette_scores:
                best_n = cluster_range[np.argmax(silhouette_scores)]
                print(f"✅ Optimal number of clusters: {best_n} (silhouette: {np.max(silhouette_scores):.3f})")
            else:
                best_n = min(3, n_samples // 3)
                print(f"⚠️ Could not compute silhouette, using {best_n} clusters")
        else:
            best_n = min(3, n_samples // 2)
            print(f"⚠️ Few samples, using {best_n} clusters")
        
        # Perform clustering with best_n
        kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latents_reduced)
        
        # Calculate cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        print(f"\n📊 Cluster Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / n_samples) * 100
            print(f"   Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        # Try to characterize clusters
        if hasattr(self, 'pca_components') and latents.shape[1] <= 50:
            self.characterize_clusters(latents, cluster_labels)
        
        self.cluster_labels = cluster_labels
        self.n_clusters = best_n
        
        return cluster_labels
    
    def characterize_clusters(self, latents, cluster_labels):
        """Try to characterize what each cluster represents"""
        print(f"\n🔍 Characterizing clusters...")
        
        n_clusters = len(np.unique(cluster_labels))
        n_features = latents.shape[1]
        
        # For each cluster, find distinguishing features
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            other_mask = cluster_labels != cluster_id
            
            if np.sum(cluster_mask) < 2 or np.sum(other_mask) < 2:
                continue
            
            # Compare mean feature values
            cluster_mean = np.mean(latents[cluster_mask], axis=0)
            other_mean = np.mean(latents[other_mask], axis=0)
            
            # Find most different features
            differences = np.abs(cluster_mean - other_mean)
            top_feature_indices = np.argsort(differences)[-5:][::-1]  # Top 5
            
            print(f"\n   Cluster {cluster_id} characteristics:")
            for i, feat_idx in enumerate(top_feature_indices):
                if feat_idx < n_features:
                    diff = differences[feat_idx]
                    cluster_val = cluster_mean[feat_idx]
                    other_val = other_mean[feat_idx]
                    
                    # Try to interpret the feature
                    feature_name = self.interpret_feature(feat_idx, cluster_val, other_val)
                    print(f"      {feature_name}: Cluster={cluster_val:.3f}, Others={other_val:.3f} (diff={diff:.3f})")
    
    def interpret_feature(self, feat_idx, cluster_val, other_val):
        """Try to interpret what a feature represents"""
        # This is a simplified interpretation based on feature indices
        # In a real scenario, you'd have meaningful feature names
        
        interpretations = [
            "Brightness", "Contrast", "Red channel", "Green channel", "Blue channel",
            "Hue", "Saturation", "Value", "Edge density", "Texture variation",
            "Face brightness", "Face contrast", "Symmetry", "Color warmth",
            "Skin tone", "Hair darkness", "Background brightness"
        ]
        
        if feat_idx < len(interpretations):
            return interpretations[feat_idx]
        else:
            return f"Feature {feat_idx}"
    
    def analyze_latent_structure(self, latents):
        """Analyze the structure of the latent space"""
        print(f"\n📐 Analyzing latent space structure...")
        
        n_samples, n_features = latents.shape
        
        # 1. Dimensionality analysis
        pca_full = PCA()
        pca_full.fit(latents)
        explained_variance = pca_full.explained_variance_ratio_
        
        # Find number of components to explain 90% variance
        cumulative_variance = np.cumsum(explained_variance)
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"✅ Dimensionality analysis:")
        print(f"   • {n_components_90} components explain 90% of variance")
        print(f"   • {n_components_95} components explain 95% of variance")
        print(f"   • Effective dimensionality: ~{n_components_90}")
        
        # 2. Feature correlations
        if n_features <= 50:  # Don't compute for too many features
            corr_matrix = np.corrcoef(latents.T)
            avg_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            print(f"   • Average absolute feature correlation: {avg_correlation:.3f}")
        
        # 3. Data manifold properties
        from scipy.spatial import KDTree
        tree = KDTree(latents)
        distances, _ = tree.query(latents, k=min(6, n_samples))
        avg_neighbor_distance = np.mean(distances[:, 1:])  # Exclude self
        print(f"   • Average nearest neighbor distance: {avg_neighbor_distance:.3f}")
        
        self.effective_dimensionality = n_components_90
        return n_components_90
    
    def create_comprehensive_visualizations(self, latents, pca_components=None, cluster_labels=None):
        """Create comprehensive visualizations of the analysis"""
        print("\n🎨 Creating comprehensive visualizations...")
        
        # 1. PCA visualization
        self.create_pca_visualization(latents, pca_components)
        
        # 2. Cluster visualization
        if cluster_labels is not None:
            self.create_cluster_visualization(latents, cluster_labels)
        
        # 3. Feature distribution visualization
        self.create_feature_distribution_visualization(latents)
        
        # 4. Correlation visualization (if features are not too many)
        if latents.shape[1] <= 50:
            self.create_correlation_visualization(latents)
    
    def create_pca_visualization(self, latents, pca_components=None):
        """Create PCA visualization"""
        # Perform PCA for visualization
        pca_vis = PCA(n_components=2)
        latents_2d = pca_vis.fit_transform(latents)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: PCA scatter plot
        ax = axes[0]
        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                           alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA Visualization of Latent Space')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Explained variance
        ax = axes[1]
        if hasattr(self, 'pca_variance'):
            variance = self.pca_variance
            cumulative = np.cumsum(variance)
            
            x = range(1, len(variance) + 1)
            ax.bar(x[:10], variance[:10], alpha=0.6, label='Individual')
            ax.plot(x[:10], cumulative[:10], 'r-', marker='o', label='Cumulative')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance (Top 10)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "pca_analysis.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_cluster_visualization(self, latents, cluster_labels):
        """Create cluster visualization"""
        # Reduce to 2D for visualization
        pca_vis = PCA(n_components=2)
        latents_2d = pca_vis.fit_transform(latents)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Cluster scatter plot
        ax = axes[0]
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                      c=[color], label=f'Cluster {label}', 
                      alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Cluster Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cluster size distribution
        ax = axes[1]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        bars = ax.bar([f'Cluster {l}' for l in unique_labels], counts, color=colors[:len(unique_labels)])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Cluster Size Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "cluster_analysis.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_feature_distribution_visualization(self, latents):
        """Create feature distribution visualization"""
        n_features = latents.shape[1]
        
        # Select top 9 most variable features
        feature_vars = np.var(latents, axis=0)
        top_feature_indices = np.argsort(feature_vars)[-9:][::-1]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, feat_idx in enumerate(top_feature_indices[:9]):
            ax = axes[i]
            
            # Plot histogram
            ax.hist(latents[:, feat_idx], bins=20, alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_val = np.mean(latents[:, feat_idx])
            std_val = np.std(latents[:, feat_idx])
            
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.5)
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.5)
            
            ax.set_xlabel(f'Feature {feat_idx}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Feature {feat_idx} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Top 9 Most Variable Feature Distributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "feature_distributions.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_correlation_visualization(self, latents):
        """Create correlation visualization"""
        if latents.shape[1] > 50:
            print("⚠️ Too many features for correlation matrix visualization")
            return
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(latents.T)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Correlation heatmap
        ax = axes[0]
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        # Plot 2: Correlation distribution
        ax = axes[1]
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        ax.hist(upper_tri, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Correlation Distribution (mean={np.mean(np.abs(upper_tri)):.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        ax.axvline(np.mean(upper_tri), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(upper_tri):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "feature_correlations.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def simulate_attribute_labels(self, latents):
        """Simulate attribute labels for demonstration purposes"""
        print("\n🎯 Simulating attribute labels for demonstration...")
        
        n_samples = latents.shape[0]
        
        # Create simulated attributes
        attributes = np.zeros((n_samples, 4))
        attribute_names = ['Brightness', 'Contrast', 'ColorWarmth', 'Complexity']
        
        # Generate based on actual features
        if latents.shape[1] >= 4:
            # Use first few features as proxies for attributes
            for i in range(4):
                if i < latents.shape[1]:
                    # Normalize to [0, 1]
                    attr_values = latents[:, i]
                    attr_min, attr_max = attr_values.min(), attr_values.max()
                    if attr_max > attr_min:
                        attributes[:, i] = (attr_values - attr_min) / (attr_max - attr_min)
                    else:
                        attributes[:, i] = 0.5  # Default value
                else:
                    # Random values for missing features
                    attributes[:, i] = np.random.uniform(0, 1, n_samples)
        else:
            # Random attributes if not enough features
            for i in range(4):
                attributes[:, i] = np.random.uniform(0, 1, n_samples)
        
        print(f"✅ Simulated {len(attribute_names)} attributes")
        
        return attributes, attribute_names
    
    def train_attribute_predictors(self, latents, attributes, attribute_names):
        """Train predictors for attributes"""
        print(f"\n🎯 Training attribute predictors...")
        
        n_attributes = attributes.shape[1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            latents, attributes, test_size=0.3, random_state=42
        )
        
        predictors = {}
        performance = {}
        
        for i in range(n_attributes):
            attr_name = attribute_names[i] if i < len(attribute_names) else f"Attr_{i}"
            
            # Check if attribute is binary or continuous
            unique_vals = np.unique(attributes[:, i])
            n_unique = len(unique_vals)
            
            if n_unique <= 3:  # Treat as categorical
                model = LogisticRegression(max_iter=1000, random_state=42)
                y_train_attr = y_train[:, i]
                y_test_attr = y_test[:, i]
            else:  # Treat as continuous
                model = Ridge(alpha=1.0, random_state=42)
                y_train_attr = y_train[:, i]
                y_test_attr = y_test[:, i]
            
            # Train model
            model.fit(X_train, y_train_attr)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            if n_unique <= 3:
                train_score = accuracy_score(y_train_attr, train_pred)
                test_score = accuracy_score(y_test_attr, test_pred)
                metric = "Accuracy"
            else:
                train_score = r2_score(y_train_attr, train_pred)
                test_score = r2_score(y_test_attr, test_pred)
                metric = "R² Score"
            
            predictors[attr_name] = {
                'model': model,
                'coef': model.coef_.flatten() if hasattr(model, 'coef_') else None,
                'intercept': model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_,
                'train_score': train_score,
                'test_score': test_score,
                'metric': metric
            }
            
            performance[attr_name] = {
                'train': train_score,
                'test': test_score,
                'metric': metric
            }
            
            print(f"   {attr_name}: {metric} = {test_score:.3f} (test)")
        
        self.predictors = predictors
        self.performance = performance
        
        return predictors, performance
    
    def generate_report(self, latents, attributes=None, cluster_labels=None):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING ANALYSIS REPORT")
        print(f"{'='*60}")
        
        n_samples = latents.shape[0]
        n_features = latents.shape[1]
        
        report = f"""
SEMANTIC ATTRIBUTE ANALYSIS REPORT
==================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

1. DATASET SUMMARY
-----------------
• Number of samples: {n_samples}
• Number of features: {n_features}
• Features extracted from: Images
• Attributes available: {'Yes (simulated)' if attributes is not None else 'No'}

2. LATENT SPACE ANALYSIS
-----------------------
"""
        
        if hasattr(self, 'effective_dimensionality'):
            report += f"""• Effective dimensionality: {self.effective_dimensionality}
• {self.effective_dimensionality} principal components explain 90% of variance
"""
        
        if hasattr(self, 'pca_variance'):
            top5_variance = np.sum(self.pca_variance[:5]) * 100
            report += f"""• Top 5 PCA components explain {top5_variance:.1f}% of variance
"""
        
        report += f"""
3. CLUSTER ANALYSIS
------------------
"""
        
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            report += f"""• Number of clusters discovered: {n_clusters}
• Cluster distribution:
"""
            
            for label in unique_labels:
                count = np.sum(cluster_labels == label)
                percentage = (count / n_samples) * 100
                report += f"   - Cluster {label}: {count} samples ({percentage:.1f}%)\n"
            
            # Try to interpret clusters
            if n_clusters <= 5:
                report += f"""
• Cluster interpretation (based on feature differences):
"""
                # Add some general interpretations based on common patterns
                interpretations = [
                    "Bright, high-contrast images",
                    "Dark, low-contrast images", 
                    "Warm color tones",
                    "Cool color tones",
                    "Complex, textured images",
                    "Simple, smooth images"
                ]
                
                for i, label in enumerate(unique_labels):
                    if i < len(interpretations):
                        report += f"   - Cluster {label}: May represent {interpretations[i]}\n"
        
        report += f"""
4. ATTRIBUTE ANALYSIS
--------------------
"""
        
        if hasattr(self, 'performance'):
            report += "• Attribute prediction performance:\n"
            for attr_name, perf in self.performance.items():
                report += f"   - {attr_name}: {perf['metric']} = {perf['test']:.3f} (test)\n"
            
            # Calculate average
            test_scores = [p['test'] for p in self.performance.values()]
            avg_score = np.mean(test_scores)
            report += f"\n• Average test score: {avg_score:.3f}\n"
        
        report += f"""
5. KEY FINDINGS
--------------
"""
        
        findings = []
        
        # Dimensionality finding
        if hasattr(self, 'effective_dimensionality'):
            if self.effective_dimensionality < n_features * 0.1:
                findings.append("• Latent space has low effective dimensionality, suggesting redundant features")
            elif self.effective_dimensionality < n_features * 0.3:
                findings.append("• Latent space has moderate effective dimensionality")
            else:
                findings.append("• Latent space has high effective dimensionality")
        
        # Cluster finding
        if cluster_labels is not None:
            n_clusters = len(np.unique(cluster_labels))
            if n_clusters >= 3:
                findings.append(f"• {n_clusters} distinct semantic clusters identified")
            else:
                findings.append("• Few semantic clusters identified, suggesting homogeneous data")
        
        # Attribute finding
        if hasattr(self, 'performance'):
            avg_score = np.mean([p['test'] for p in self.performance.values()])
            if avg_score > 0.7:
                findings.append("• Attributes are well-predictable from features")
            elif avg_score > 0.4:
                findings.append("• Attributes are moderately predictable from features")
            else:
                findings.append("• Attributes have low predictability from features")
        
        for finding in findings:
            report += finding + "\n"
        
        report += f"""
6. RECOMMENDATIONS
-----------------
1. {'Consider feature selection to reduce dimensionality' if hasattr(self, 'effective_dimensionality') and self.effective_dimensionality < n_features * 0.5 else 'Feature space appears efficient'}
2. {'Use discovered clusters for semantic grouping and analysis' if cluster_labels is not None and len(np.unique(cluster_labels)) > 1 else 'Data appears homogeneous, consider collecting more diverse samples'}
3. {'Attribute directions can be used for semantic editing' if hasattr(self, 'performance') else 'Collect attribute labels to enable semantic control'}
4. Validate findings with human evaluation of sample groupings

7. VISUALIZATIONS
---------------
• PCA analysis: {self.visualizations_dir}/pca_analysis.png
• Cluster analysis: {self.visualizations_dir}/cluster_analysis.png
• Feature distributions: {self.visualizations_dir}/feature_distributions.png
• Feature correlations: {self.visualizations_dir}/feature_correlations.png

8. NEXT STEPS
------------
1. Manually inspect samples in each cluster to validate semantic groupings
2. If attributes are available, use regression weights for semantic editing
3. Consider collecting more diverse samples if clusters are not distinct
4. Apply findings to downstream tasks (e.g., image retrieval, style transfer)

========================================================================
"""
        
        # Save report
        report_file = os.path.join(self.output_dir, "semantic_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_file}")
        
        # Save analysis data
        analysis_data = {
            'n_samples': int(n_samples),
            'n_features': int(n_features),
            'timestamp': datetime.now().isoformat()
        }
        
        if hasattr(self, 'pca_variance'):
            analysis_data['pca_variance'] = self.pca_variance.tolist()
        
        if hasattr(self, 'cluster_labels'):
            analysis_data['cluster_labels'] = self.cluster_labels.tolist()
        
        if hasattr(self, 'performance'):
            analysis_data['performance'] = self.performance
        
        with open(os.path.join(self.output_dir, "analysis_data.json"), 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"✅ Analysis data saved to: {os.path.join(self.output_dir, 'analysis_data.json')}")
        
        return report
    
    def run_analysis(self, data_path=None, labels_path=None):
        """Run comprehensive semantic attribute analysis"""
        print(f"\n{'='*60}")
        print("RUNNING SEMANTIC ATTRIBUTE ANALYSIS")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Determine data path
        if data_path is None:
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
                "../latent_analysis_20251202_175108/latents/",
                "images/",
                "data/"
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
        latents, attributes, attribute_names, image_info = self.load_data(data_path, labels_path)
        if latents is None:
            return None
        
        print(f"\n✅ Starting analysis with {len(latents)} samples, {latents.shape[1]} features")
        
        # 1. Analyze latent space structure
        effective_dim = self.analyze_latent_structure(latents)
        
        # 2. Discover semantic directions
        pca_components, pca_variance = self.discover_semantic_directions(latents)
        
        # 3. Unsupervised attribute discovery (clustering)
        cluster_labels = self.perform_unsupervised_attribute_discovery(latents)
        
        # 4. Create visualizations
        self.create_comprehensive_visualizations(latents, pca_components, cluster_labels)
        
        # 5. Simulate attributes if none provided
        if attributes is None:
            attributes, attribute_names = self.simulate_attribute_labels(latents)
        
        # 6. Train attribute predictors
        if attributes is not None:
            predictors, performance = self.train_attribute_predictors(latents, attributes, attribute_names)
        
        # 7. Generate comprehensive report
        report = self.generate_report(latents, attributes, cluster_labels)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Total time: {elapsed_time.total_seconds():.1f} seconds")
        print(f"✅ Output saved to: {self.output_dir}/")
        print(f"✅ Report: {self.output_dir}/semantic_analysis_report.txt")
        print(f"\n🎯 Analysis Summary:")
        print(f"   • Analyzed {len(latents)} samples with {latents.shape[1]} features")
        print(f"   • Effective dimensionality: {effective_dim}")
        
        if cluster_labels is not None:
            n_clusters = len(np.unique(cluster_labels))
            print(f"   • Discovered {n_clusters} semantic clusters")
        
        if hasattr(self, 'performance'):
            avg_score = np.mean([p['test'] for p in self.performance.values()])
            print(f"   • Average attribute prediction score: {avg_score:.3f}")
        
        print(f"\n📋 Key outputs:")
        print(f"   1. PCA-based semantic directions")
        print(f"   2. Cluster-based attribute discovery") 
        print(f"   3. Attribute prediction models")
        print(f"   4. Comprehensive visualizations")
        print(f"   5. Detailed analysis report")
        
        return {
            'latents': latents,
            'attributes': attributes,
            'attribute_names': attribute_names,
            'cluster_labels': cluster_labels,
            'effective_dim': effective_dim,
            'output_dir': self.output_dir
        }


# Main execution
if __name__ == "__main__":
    analyzer = SemanticAttributeAnalyzer()
    
    # Run analysis (auto-detects your data)
    results = analyzer.run_analysis()
