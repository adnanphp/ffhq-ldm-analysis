# ffhq_advanced_diversity_analysis.py
import torch
import numpy as np
import os
import sys
import json
import time
import pickle
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, ndimage
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# Add the latent-diffusion directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class AdvancedDiversityAnalyzer:
    def __init__(self, existing_dir=None):
        """Initialize the advanced diversity analyzer"""
        print(" ADVANCED DIVERSITY ANALYSIS")
        print("=" * 60)
        
        if existing_dir and os.path.exists(existing_dir):
            self.output_dir = existing_dir
            print(f" Using existing directory: {self.output_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"advanced_diversity_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f" Created new directory: {self.output_dir}")
        
        # Create subdirectories
        self.style_dir = os.path.join(self.output_dir, "style_analysis")
        self.coverage_dir = os.path.join(self.output_dir, "coverage_analysis")
        self.novelty_dir = os.path.join(self.output_dir, "novelty_analysis")
        
        for dir_path in [self.style_dir, self.coverage_dir, self.novelty_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
    
    def load_existing_faces(self, source_dir=None):
        """Load existing generated faces for analysis"""
        print(f"\n{'='*60}")
        print("LOADING EXISTING FACES FOR ANALYSIS")
        print(f"{'='*60}")
        
        import glob
        
        if source_dir is None:
            # Find the most recent diversity analysis directory
            diversity_dirs = sorted(glob.glob("diversity_analysis_*"))
            if diversity_dirs:
                source_dir = diversity_dirs[-1]
            else:
                # Fall back to quantitative evaluation directory
                quant_dirs = sorted(glob.glob("quantitative_eval_*"))
                if quant_dirs:
                    source_dir = quant_dirs[-1]
                else:
                    print("❌ No existing directories found!")
                    return None, None
        
        print(f" Loading faces from: {source_dir}")
        
        # Try different possible locations
        possible_locations = [
            os.path.join(source_dir, "sample_faces"),
            os.path.join(source_dir, "fake_samples"),
            source_dir  # Check the directory itself
        ]
        
        image_files = []
        for location in possible_locations:
            if os.path.exists(location):
                png_files = glob.glob(os.path.join(location, "*.png"))
                if png_files:
                    image_files = png_files
                    print(f"✅ Found {len(image_files)} images in: {location}")
                    break
        
        if not image_files:
            print("❌ No PNG files found!")
            return None, None
        
        # Load images
        faces = []
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        for i, img_file in enumerate(image_files[:100]):  # Limit to 100 for speed
            try:
                img = Image.open(img_file).convert('RGB')
                img_tensor = transform(img)
                img_np = img_tensor.numpy().transpose(1, 2, 0)
                
                face_data = {
                    'id': i,
                    'filename': os.path.basename(img_file),
                    'filepath': img_file,
                    'image': img_tensor,
                    'image_np': img_np,
                    'shape': img_np.shape
                }
                
                faces.append(face_data)
                
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
        
        print(f" Successfully loaded {len(faces)} faces")
        
        # Save a summary of loaded faces
        self.save_face_summary(faces)
        
        return faces, source_dir
    
    def save_face_summary(self, faces):
        """Save summary of loaded faces"""
        summary = {
            'total_faces': len(faces),
            'image_shapes': [list(f['shape']) for f in faces],
            'filenames': [f['filename'] for f in faces],
            'loaded_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "loaded_faces_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    
    def analyze_style_variation(self, faces):
        """
        Analyze style variation: pose, lighting, and expression diversity
        """
        print(f"\n{'='*60}")
        print("ANALYZING STYLE VARIATION")
        print(f"{'='*60}")
        
        print(" Extracting style features...")
        
        style_features = []
        style_metrics = []
        
        for face in faces:
            img_np = face['image_np']
            
            # 1. Lighting analysis (brightness and contrast)
            brightness = img_np.mean()
            contrast = img_np.std()
            
            # 2. Color temperature (warm vs cool)
            r, g, b = img_np.mean(axis=(0, 1))
            color_temp = r / (g + 1e-8)  # Red-green ratio
            
            # 3. Symmetry analysis (pose estimation)
            gray = np.mean(img_np, axis=2)
            height, width = gray.shape
            
            # Split image into left and right halves
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            # Flip right half for comparison
            right_half_flipped = np.fliplr(right_half)
            
            # Calculate symmetry score
            symmetry_score = 1.0 - np.abs(left_half - right_half_flipped).mean()
            
            # 4. Expression analysis (mouth region)
            # Assume face is centered, mouth is in lower middle
            mouth_region = img_np[height//2:, width//4:3*width//4, :]
            mouth_brightness = mouth_region.mean()
            
            # Smile detection (simplified)
            # Bright mouth often indicates smile
            smile_score = min(mouth_brightness / 0.5, 1.0)
            
            # 5. Head pose estimation (simplified)
            # Using face aspect ratio as proxy
            face_aspect_ratio = height / width
            
            # 6. Sharpness/focus
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            sharpness = np.hypot(sobel_x, sobel_y).mean()
            
            # Compile features
            features = np.array([
                brightness, contrast, color_temp, symmetry_score,
                smile_score, face_aspect_ratio, sharpness
            ])
            
            style_features.append(features)
            
            metrics = {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'color_temperature': float(color_temp),
                'symmetry': float(symmetry_score),
                'smile_intensity': float(smile_score),
                'aspect_ratio': float(face_aspect_ratio),
                'sharpness': float(sharpness)
            }
            
            style_metrics.append(metrics)
        
        style_features = np.array(style_features)
        
        print(f"📊 Extracted {style_features.shape[1]} style features from {len(faces)} faces")
        
        # Calculate style diversity metrics
        style_diversity = self.calculate_style_diversity(style_features)
        
        # Create style clusters
        style_clusters = self.analyze_style_clusters(style_features, faces)
        
        # Create visualizations
        self.create_style_visualizations(style_features, style_metrics, faces, style_diversity, style_clusters)
        
        # Save results
        style_results = {
            'style_diversity': style_diversity,
            'avg_style_metrics': {
                'brightness': float(np.mean([m['brightness'] for m in style_metrics])),
                'contrast': float(np.mean([m['contrast'] for m in style_metrics])),
                'symmetry': float(np.mean([m['symmetry'] for m in style_metrics])),
                'smile_intensity': float(np.mean([m['smile_intensity'] for m in style_metrics]))
            },
            'style_clusters': style_clusters,
            'n_faces': len(faces),
            'n_style_features': style_features.shape[1]
        }
        
        with open(os.path.join(self.style_dir, "style_analysis_results.json"), 'w') as f:
            json.dump(style_results, f, indent=2)
        
        print(f"\n STYLE DIVERSITY RESULTS:")
        print(f"  Style Variation Score: {style_diversity['variation_score']:.3f}")
        print(f"  Pose Diversity: {style_diversity['pose_diversity']:.3f}")
        print(f"  Lighting Diversity: {style_diversity['lighting_diversity']:.3f}")
        print(f"  Expression Diversity: {style_diversity['expression_diversity']:.3f}")
        
        if style_diversity['overall_style_score'] > 0.7:
            interpretation = "EXCELLENT style diversity"
        elif style_diversity['overall_style_score'] > 0.5:
            interpretation = "GOOD style diversity"
        elif style_diversity['overall_style_score'] > 0.3:
            interpretation = "MODERATE style diversity"
        else:
            interpretation = "LOW style diversity"
        
        print(f"📈 Overall Style Score: {style_diversity['overall_style_score']:.3f} ({interpretation})")
        
        return style_results
    
    def calculate_style_diversity(self, style_features):
        """Calculate style diversity metrics"""
        # Calculate coefficient of variation for each style dimension
        cv_scores = []
        for i in range(style_features.shape[1]):
            feature = style_features[:, i]
            if feature.mean() > 0:
                cv = feature.std() / feature.mean()
            else:
                cv = feature.std()
            cv_scores.append(cv)
        
        # Normalize CV scores to [0, 1]
        cv_scores = np.array(cv_scores)
        cv_scores = cv_scores / (cv_scores.max() + 1e-8)
        
        # Define which features correspond to which aspects
        # [brightness, contrast, color_temp, symmetry, smile, aspect_ratio, sharpness]
        lighting_indices = [0, 1, 2]  # brightness, contrast, color_temp
        pose_indices = [3, 5]  # symmetry, aspect_ratio
        expression_indices = [4]  # smile intensity
        
        lighting_diversity = cv_scores[lighting_indices].mean()
        pose_diversity = cv_scores[pose_indices].mean()
        expression_diversity = cv_scores[expression_indices].mean()
        
        # Overall variation score (higher = more diverse)
        variation_score = cv_scores.mean()
        
        # Calculate pairwise style distances
        style_distances = pairwise_distances(style_features, metric='euclidean')
        avg_style_distance = style_distances.mean()
        max_style_distance = style_distances.max()
        
        # Style coverage (how much of possible style space is covered)
        if max_style_distance > 0:
            style_coverage = avg_style_distance / max_style_distance
        else:
            style_coverage = 0
        
        # Overall style diversity score
        overall_score = (variation_score * 0.4 + 
                        style_coverage * 0.3 + 
                        (lighting_diversity + pose_diversity + expression_diversity) / 3 * 0.3)
        
        return {
            'variation_score': float(variation_score),
            'lighting_diversity': float(lighting_diversity),
            'pose_diversity': float(pose_diversity),
            'expression_diversity': float(expression_diversity),
            'avg_style_distance': float(avg_style_distance),
            'max_style_distance': float(max_style_distance),
            'style_coverage': float(style_coverage),
            'overall_style_score': float(overall_score)
        }
    
    def analyze_style_clusters(self, style_features, faces, n_clusters=4):
        """Analyze style clusters using K-means"""
        from sklearn.cluster import KMeans
        
        print(" Analyzing style clusters...")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(style_features)
        
        # Analyze cluster characteristics
        clusters = {}
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_faces = [faces[i] for i in np.where(mask)[0]]
            cluster_features = style_features[mask]
            
            # Calculate cluster centroids
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Determine cluster type based on centroid
            # [brightness, contrast, color_temp, symmetry, smile, aspect_ratio, sharpness]
            if centroid[4] > 0.6:  # High smile intensity
                cluster_type = "Smiling"
            elif centroid[0] > 0.6:  # High brightness
                cluster_type = "Well-lit"
            elif centroid[1] > 0.5:  # High contrast
                cluster_type = "High-contrast"
            elif centroid[3] > 0.7:  # High symmetry
                cluster_type = "Front-facing"
            else:
                cluster_type = "Neutral"
            
            clusters[cluster_id] = {
                'type': cluster_type,
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(faces) * 100),
                'centroid': centroid.tolist(),
                'face_ids': [faces[i]['id'] for i in np.where(mask)[0]]
            }
        
        print(f"📊 Found {n_clusters} style clusters:")
        for cluster_id, info in clusters.items():
            print(f"  Cluster {cluster_id} ({info['type']}): {info['size']} faces ({info['percentage']:.1f}%)")
        
        return clusters
    
    def create_style_visualizations(self, style_features, style_metrics, faces, style_diversity, style_clusters):
        """Create visualizations for style analysis"""
        print(" Creating style visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Style features radar chart
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        
        feature_names = ['Brightness', 'Contrast', 'Color Temp', 'Symmetry', 
                        'Smile', 'Aspect Ratio', 'Sharpness']
        
        # Calculate average feature values
        avg_features = np.mean(style_features, axis=0)
        
        # Normalize for radar chart
        avg_normalized = avg_features / (avg_features.max() + 1e-8)
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        avg_normalized = np.concatenate([avg_normalized, [avg_normalized[0]]])
        angles += angles[:1]
        
        ax1.plot(angles, avg_normalized, 'o-', linewidth=2, markersize=8)
        ax1.fill(angles, avg_normalized, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(feature_names)
        ax1.set_ylim([0, 1])
        ax1.set_title('Average Style Features', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. Style diversity scores
        ax2 = plt.subplot(2, 3, 2)
        
        diversity_categories = ['Variation', 'Lighting', 'Pose', 'Expression', 'Coverage']
        diversity_scores = [
            style_diversity['variation_score'],
            style_diversity['lighting_diversity'],
            style_diversity['pose_diversity'],
            style_diversity['expression_diversity'],
            style_diversity['style_coverage']
        ]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(diversity_categories)))
        bars = ax2.bar(diversity_categories, diversity_scores, color=colors, alpha=0.8)
        ax2.set_ylabel('Score (0-1, higher is better)', fontsize=12)
        ax2.set_title('Style Diversity Scores', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        
        # Add value labels
        for bar, score in zip(bars, diversity_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cluster visualization
        ax3 = plt.subplot(2, 3, 3)
        
        # Apply PCA for 2D visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        style_2d = pca.fit_transform(style_features)
        
        # Get cluster labels
        cluster_labels = []
        for face_id in range(len(faces)):
            for cluster_id, cluster_info in style_clusters.items():
                if face_id in cluster_info['face_ids']:
                    cluster_labels.append(cluster_id)
                    break
        
        # Plot clusters
        unique_clusters = sorted(set(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_labels) == cluster_id
            cluster_name = style_clusters[cluster_id]['type']
            ax3.scatter(style_2d[mask, 0], style_2d[mask, 1],
                       c=[colors[cluster_id]], label=f'{cluster_name} ({np.sum(mask)})',
                       alpha=0.7, s=60)
        
        ax3.set_xlabel('PCA Component 1', fontsize=12)
        ax3.set_ylabel('PCA Component 2', fontsize=12)
        ax3.set_title(f'Style Clusters (Explained variance: {pca.explained_variance_ratio_.sum():.2f})',
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Example faces from each cluster
        ax4 = plt.subplot(2, 3, 4)
        
        # Show one example from each cluster
        examples_per_cluster = 1
        total_examples = min(4, len(style_clusters))  # Max 4 clusters shown
        
        # Create a grid for examples
        example_images = []
        example_titles = []
        
        for i, (cluster_id, cluster_info) in enumerate(list(style_clusters.items())[:total_examples]):
            if cluster_info['face_ids']:
                face_id = cluster_info['face_ids'][0]
                face_img = faces[face_id]['image_np']
                example_images.append(face_img)
                example_titles.append(f"Cluster {cluster_id}\n{cluster_info['type']}")
        
        # Create montage
        if example_images:
            n_cols = 2
            n_rows = (len(example_images) + n_cols - 1) // n_cols
            
            # Clear axis and create grid
            ax4.clear()
            ax4.axis('off')
            
            # Create grid for displaying images
            grid = ax4
            grid.axis('off')
            
            # We'll handle this differently - create actual subplots
            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(10, 8))
            if n_rows == 1 and n_cols == 1:
                axes2 = np.array([[axes2]])
            elif n_rows == 1:
                axes2 = axes2.reshape(1, -1)
            
            for idx, (img, title) in enumerate(zip(example_images, example_titles)):
                row = idx // n_cols
                col = idx % n_cols
                axes2[row, col].imshow(img)
                axes2[row, col].set_title(title, fontsize=10)
                axes2[row, col].axis('off')
            
            # Hide unused subplots
            for idx in range(len(example_images), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes2[row, col].axis('off')
            
            fig2.suptitle('Example Faces from Each Style Cluster', fontsize=14, fontweight='bold')
            fig2.tight_layout()
            
            # Save separately
            cluster_examples_path = os.path.join(self.style_dir, "style_cluster_examples.png")
            fig2.savefig(cluster_examples_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            # For the main figure, just add a placeholder
            ax4.text(0.5, 0.5, f'Style Cluster Examples\nSaved separately\nSee: {cluster_examples_path}',
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Style Cluster Examples', fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        # 5. Style distribution histograms
        ax5 = plt.subplot(2, 3, 5)
        
        # Plot distribution of key style features
        key_features = ['Brightness', 'Contrast', 'Smile Intensity']
        feature_indices = [0, 1, 4]
        
        for i, (feature_name, idx) in enumerate(zip(key_features, feature_indices)):
            values = style_features[:, idx]
            ax5.hist(values, bins=20, alpha=0.5, label=feature_name, density=True)
        
        ax5.set_xlabel('Feature Value', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.set_title('Style Feature Distributions', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Overall style score gauge
        ax6 = plt.subplot(2, 3, 6)
        
        overall_score = style_diversity['overall_style_score']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background
        ax6.plot(theta, r, color='gray', linewidth=10, alpha=0.3)
        
        # Score arc
        score_angle = overall_score * np.pi
        score_theta = np.linspace(0, score_angle, 100)
        score_r = np.ones_like(score_theta)
        
        # Color based on score
        if overall_score > 0.7:
            gauge_color = 'green'
        elif overall_score > 0.5:
            gauge_color = 'orange'
        else:
            gauge_color = 'red'
        
        ax6.plot(score_theta, score_r, color=gauge_color, linewidth=10)
        
        # Add needle
        needle_angle = overall_score * np.pi
        ax6.plot([needle_angle, needle_angle], [0.7, 1.1], color='black', linewidth=3)
        
        # Labels
        ax6.text(np.pi/2, 0.5, f'{overall_score:.3f}', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        if overall_score > 0.7:
            rating = "EXCELLENT"
        elif overall_score > 0.5:
            rating = "GOOD"
        elif overall_score > 0.3:
            rating = "MODERATE"
        else:
            rating = "LOW"
        
        ax6.text(np.pi/2, 0.3, rating, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax6.set_xlim([0, np.pi])
        ax6.set_ylim([0, 1.2])
        ax6.axis('off')
        ax6.set_title('Overall Style Score', fontsize=14, fontweight='bold')
        
        # Main title
        plt.suptitle('FFHQ Generated Faces - Advanced Style Analysis', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save main figure
        style_viz_path = os.path.join(self.style_dir, "style_analysis_visualization.png")
        plt.savefig(style_viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Style analysis visualizations saved in: {self.style_dir}/")
        
        return style_viz_path
    
    def analyze_mode_coverage(self, faces, reference_features=None, k=3):
        """
        Analyze mode coverage using precision-recall metrics
        Measures how well generated distribution covers real distribution
        """
        print(f"\n{'='*60}")
        print("ANALYZING MODE COVERAGE")
        print(f"{'='*60}")
        
        print(" Calculating mode coverage metrics...")
        
        # Extract features from generated faces
        gen_features = []
        for face in faces:
            img_np = face['image_np']
            
            # Use simple color and texture features
            color_features = img_np.mean(axis=(0, 1)).flatten()
            texture_features = np.array([img_np.std()])
            
            # Add style features if available
            gray = np.mean(img_np, axis=2)
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.hypot(sobel_x, sobel_y)
            edge_features = np.array([edge_magnitude.mean(), edge_magnitude.std()])
            
            features = np.concatenate([color_features, texture_features, edge_features])
            gen_features.append(features)
        
        gen_features = np.array(gen_features)
        
        # Create simulated reference features (real distribution)
        if reference_features is None:
            print("💡 Creating simulated real distribution...")
            
            # Create reference features with more diversity
            n_reference = min(200, len(faces) * 3)
            
            # Sample from broader distribution
            reference_mean = np.random.randn(gen_features.shape[1]) * 0.3
            reference_cov = np.eye(gen_features.shape[1]) * 0.4
            
            reference_features = np.random.multivariate_normal(
                reference_mean, reference_cov, size=n_reference
            )
        
        # Calculate precision and recall
        precision, recall, density, coverage = self.calculate_precision_recall_advanced(
            reference_features, gen_features, k=k
        )
        
        # Calculate additional coverage metrics
        coverage_metrics = self.calculate_coverage_metrics(reference_features, gen_features)
        
        print(f"\n MODE COVERAGE METRICS:")
        print(f"  Precision: {precision:.3f} (quality of generated samples)")
        print(f"  Recall: {recall:.3f} (coverage of real distribution)")
        print(f"  F1 Score: {2 * precision * recall / (precision + recall + 1e-8):.3f}")
        print(f"  Density: {density:.3f} (sample quality within manifolds)")
        print(f"  Coverage: {coverage:.3f} (sample diversity)")
        
        # Create visualizations
        self.create_coverage_visualizations(reference_features, gen_features, 
                                          precision, recall, density, coverage, coverage_metrics)
        
        # Save results
        coverage_results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(2 * precision * recall / (precision + recall + 1e-8)),
            'density': float(density),
            'coverage': float(coverage),
            'coverage_metrics': coverage_metrics,
            'n_generated': len(faces),
            'n_reference': len(reference_features),
            'k_neighbors': k
        }
        
        with open(os.path.join(self.coverage_dir, "coverage_analysis_results.json"), 'w') as f:
            json.dump(coverage_results, f, indent=2)
        
        return coverage_results
    
    def calculate_precision_recall_advanced(self, real_features, gen_features, k=3):
        """Calculate advanced precision and recall metrics"""
        from sklearn.neighbors import NearestNeighbors
        
        # Fit k-NN models
        knn_real = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn_real.fit(real_features)
        
        knn_gen = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn_gen.fit(gen_features)
        
        # Calculate distances
        distances_gen_to_real, _ = knn_real.kneighbors(gen_features)
        distances_real_to_gen, _ = knn_gen.kneighbors(real_features)
        
        # Calculate manifold boundaries
        real_manifold = np.percentile(distances_real_to_gen[:, -1], 95)
        gen_manifold = np.percentile(distances_gen_to_real[:, -1], 95)
        
        # Precision: fraction of generated samples within real manifold
        precision = np.mean(distances_gen_to_real[:, -1] <= real_manifold)
        
        # Recall: fraction of real samples within generated manifold
        recall = np.mean(distances_real_to_gen[:, -1] <= gen_manifold)
        
        # Density: average inverse distance within manifolds
        density_real = np.mean(1.0 / (distances_real_to_gen[:, -1] + 1e-8))
        density_gen = np.mean(1.0 / (distances_gen_to_real[:, -1] + 1e-8))
        density = (density_real + density_gen) / 2
        
        # Normalize density
        density = min(density / 10, 1.0)  # Rough normalization
        
        # Coverage: how spread out are the generated samples
        gen_distances = pairwise_distances(gen_features, metric='euclidean')
        max_gen_distance = gen_distances.max()
        
        # Compare to real distribution spread
        real_distances = pairwise_distances(real_features, metric='euclidean')
        max_real_distance = real_distances.max()
        
        if max_real_distance > 0:
            coverage = max_gen_distance / max_real_distance
        else:
            coverage = 0
        
        coverage = min(coverage, 1.0)  # Cap at 1.0
        
        return precision, recall, density, coverage
    
    def calculate_coverage_metrics(self, real_features, gen_features):
        """Calculate additional coverage metrics"""
        # Calculate centroids
        real_centroid = real_features.mean(axis=0)
        gen_centroid = gen_features.mean(axis=0)
        
        # Calculate spread (standard deviation)
        real_spread = real_features.std(axis=0).mean()
        gen_spread = gen_features.std(axis=0).mean()
        
        # Calculate overlap score (simplified)
        from scipy.stats import gaussian_kde
        
        # Project to 1D for simplicity
        real_1d = real_features.mean(axis=1)
        gen_1d = gen_features.mean(axis=1)
        
        # Estimate densities
        kde_real = gaussian_kde(real_1d)
        kde_gen = gaussian_kde(gen_1d)
        
        # Evaluate on common grid
        x_min = min(real_1d.min(), gen_1d.min())
        x_max = max(real_1d.max(), gen_1d.max())
        x_grid = np.linspace(x_min, x_max, 100)
        
        pdf_real = kde_real(x_grid)
        pdf_gen = kde_gen(x_grid)
        
        # Calculate overlap (intersection over union)
        intersection = np.minimum(pdf_real, pdf_gen).sum()
        union = np.maximum(pdf_real, pdf_gen).sum()
        
        if union > 0:
            overlap_score = intersection / union
        else:
            overlap_score = 0
        
        return {
            'centroid_distance': float(np.linalg.norm(real_centroid - gen_centroid)),
            'real_spread': float(real_spread),
            'gen_spread': float(gen_spread),
            'spread_ratio': float(gen_spread / (real_spread + 1e-8)),
            'overlap_score': float(overlap_score)
        }
    
    def create_coverage_visualizations(self, real_features, gen_features, 
                                     precision, recall, density, coverage, coverage_metrics):
        """Create visualizations for mode coverage analysis"""
        print("📊 Creating coverage visualizations...")
        
        # Apply dimensionality reduction for visualization
        from sklearn.decomposition import PCA
        
        combined_features = np.vstack([real_features, gen_features])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined_features)
        
        n_real = len(real_features)
        n_gen = len(gen_features)
        
        real_2d = combined_2d[:n_real]
        gen_2d = combined_2d[n_real:]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Feature space visualization
        ax1.scatter(real_2d[:, 0], real_2d[:, 1], 
                   c='blue', alpha=0.5, s=30, label='Real (simulated)')
        ax1.scatter(gen_2d[:, 0], gen_2d[:, 1], 
                   c='red', alpha=0.7, s=50, label='Generated', marker='^')
        
        ax1.set_xlabel('PCA Component 1', fontsize=12)
        ax1.set_ylabel('PCA Component 2', fontsize=12)
        ax1.set_title('Feature Space Coverage', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall visualization
        ax2.bar(['Precision', 'Recall', 'F1'], 
               [precision, recall, 2 * precision * recall / (precision + recall + 1e-8)],
               color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Precision-Recall Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (label, value) in enumerate(zip(['Precision', 'Recall', 'F1'], 
                                              [precision, recall, 
                                               2 * precision * recall / (precision + recall + 1e-8)])):
            ax2.text(i, value + 0.02, f'{value:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Density and coverage visualization
        ax3.bar(['Density', 'Coverage'], 
               [density, coverage],
               color=['orange', 'purple'], alpha=0.8)
        
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Density and Coverage', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (label, value) in enumerate(zip(['Density', 'Coverage'], [density, coverage])):
            ax3.text(i, value + 0.02, f'{value:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Coverage metrics radar chart
        ax4 = plt.subplot(224, projection='polar')
        
        metrics_categories = ['Centroid Dist', 'Spread Ratio', 'Overlap']
        metrics_values = [
            min(coverage_metrics['centroid_distance'] / 5, 1.0),  # Normalized
            min(coverage_metrics['spread_ratio'], 2.0) / 2.0,  # Normalized
            coverage_metrics['overlap_score']
        ]
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(metrics_categories), endpoint=False).tolist()
        metrics_values = metrics_values + [metrics_values[0]]
        angles += angles[:1]
        
        ax4.plot(angles, metrics_values, 'o-', linewidth=2, markersize=8)
        ax4.fill(angles, metrics_values, alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics_categories)
        ax4.set_ylim([0, 1])
        ax4.set_title('Additional Coverage Metrics', fontsize=14, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Mode Coverage Analysis', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        coverage_viz_path = os.path.join(self.coverage_dir, "coverage_analysis_visualization.png")
        plt.savefig(coverage_viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Coverage analysis visualizations saved: {coverage_viz_path}")
        
        return coverage_viz_path
    
# Fixed novelty assessment code (replace from line 917 onward)

    def analyze_novelty_assessment(self, faces, memorization_threshold=0.1):
        """
        Analyze novelty: percentage of unique, non-memorized generations
        """
        print(f"\n{'='*60}")
        print("ANALYZING NOVELTY ASSESSMENT")
        print(f"{'='*60}")
        
        print("🔍 Calculating novelty metrics...")
        
        # Extract features for comparison - SIMPLIFIED VERSION
        features = []
        for face in faces:
            img_np = face['image_np']
            
            # Use simple feature vector instead of perceptual hash
            # 1. Color features
            color_features = img_np.mean(axis=(0, 1)).flatten()
            
            # 2. Texture features
            gray = np.mean(img_np, axis=2)
            contrast = gray.std()
            
            # 3. Shape features (aspect ratio)
            aspect_ratio = img_np.shape[0] / img_np.shape[1]
            
            # 4. Edge features
            from scipy import ndimage
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.hypot(sobel_x, sobel_y)
            edge_mean = edge_magnitude.mean()
            edge_std = edge_magnitude.std()
            
            # Combine all features
            feature_vector = np.concatenate([
                color_features,
                np.array([contrast, aspect_ratio, edge_mean, edge_std])
            ])
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Calculate pairwise similarities using cosine similarity
        similarity_matrix = self.calculate_similarity_matrix_fixed(features)
        
        # Identify potential memorizations (very similar pairs)
        memorization_mask = similarity_matrix > (1.0 - memorization_threshold)
        
        # Exclude self-similarity
        np.fill_diagonal(memorization_mask, False)
        
        # Count unique vs potentially memorized
        n_faces = len(faces)
        n_potential_memorizations = np.sum(memorization_mask) // 2  # Divide by 2 for pairs
        
        # Identify clusters of similar faces
        clusters = self.identify_similarity_clusters(similarity_matrix, threshold=0.8)
        
        # Calculate novelty metrics
        novelty_metrics = self.calculate_novelty_metrics(faces, similarity_matrix, clusters)
        
        print(f"\n NOVELTY METRICS:")
        print(f"  Total faces analyzed: {n_faces}")
        print(f"  Potential memorizations: {n_potential_memorizations}")
        print(f"  Unique faces: {n_faces - n_potential_memorizations}")
        print(f"  Novelty percentage: {novelty_metrics['novelty_percentage']:.1f}%")
        print(f"  Average similarity: {novelty_metrics['avg_similarity']:.3f}")
        print(f"  Max similarity: {novelty_metrics['max_similarity']:.3f}")
        print(f"  Similarity clusters: {novelty_metrics['n_clusters']}")
        
        # Create visualizations
        self.create_novelty_visualizations(faces, similarity_matrix, clusters, novelty_metrics)
        
        # Save results
        novelty_results = {
            'total_faces': n_faces,
            'potential_memorizations': int(n_potential_memorizations),
            'unique_faces': int(n_faces - n_potential_memorizations),
            'novelty_percentage': float(novelty_metrics['novelty_percentage']),
            'novelty_metrics': novelty_metrics,
            'memorization_threshold': float(memorization_threshold),
            'similarity_clusters': clusters
        }
        
        with open(os.path.join(self.novelty_dir, "novelty_analysis_results.json"), 'w') as f:
            json.dump(novelty_results, f, indent=2)
        
        # Interpretation
        if novelty_metrics['novelty_percentage'] > 90:
            interpretation = "EXCELLENT novelty (highly unique generations)"
        elif novelty_metrics['novelty_percentage'] > 70:
            interpretation = "GOOD novelty"
        elif novelty_metrics['novelty_percentage'] > 50:
            interpretation = "MODERATE novelty (some repetition)"
        else:
            interpretation = "LOW novelty (significant memorization)"
        
        print(f" Interpretation: {interpretation}")
        
        return novelty_results
    
    def calculate_similarity_matrix_fixed(self, features):
        """Calculate similarity matrix using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(features)
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Clip to valid range [0, 1]
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix
    
    def identify_similarity_clusters(self, similarity_matrix, threshold=0.8):
        """Identify clusters of similar faces"""
        from sklearn.cluster import DBSCAN
        
        # Use DBSCAN for density-based clustering
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Set epsilon based on threshold (1 - threshold)
        eps = 1 - threshold
        
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        # Organize clusters
        clusters = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Noise points (unique faces)
            
            indices = np.where(labels == label)[0]
            cluster_size = len(indices)
            
            # Calculate cluster similarity
            if cluster_size > 1:
                cluster_similarities = similarity_matrix[indices][:, indices]
                avg_similarity = cluster_similarities.mean()
            else:
                avg_similarity = 1.0
            
            clusters[label] = {
                'size': int(cluster_size),
                'indices': indices.tolist(),
                'avg_similarity': float(avg_similarity)
            }
        
        # Also count unique faces (noise points)
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            # Create a singleton cluster for each unique face
            cluster_id = len(clusters)  # Use new ID
            clusters[cluster_id] = {
                'size': 1,
                'indices': [int(idx)],
                'avg_similarity': 1.0
            }
        
        return clusters
    
    def calculate_novelty_metrics(self, faces, similarity_matrix, clusters):
        """Calculate comprehensive novelty metrics"""
        n_faces = len(faces)
        
        # Calculate average similarity (excluding self-similarity)
        mask = ~np.eye(n_faces, dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        
        # Maximum similarity (excluding self-similarity)
        max_similarity = similarity_matrix[mask].max()
        
        # Count singleton clusters (clusters of size 1)
        n_singletons = sum(1 for cluster in clusters.values() if cluster['size'] == 1)
        novelty_percentage = (n_singletons / n_faces) * 100 if n_faces > 0 else 0
        
        # Calculate diversity within multi-face clusters
        cluster_diversity = []
        for cluster in clusters.values():
            if cluster['size'] > 1:
                diversity = 1.0 - cluster['avg_similarity']
                cluster_diversity.append(diversity)
        
        avg_cluster_diversity = np.mean(cluster_diversity) if cluster_diversity else 1.0
        
        return {
            'avg_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'novelty_percentage': float(novelty_percentage),
            'n_clusters': len(clusters),
            'n_singletons': int(n_singletons),
            'avg_cluster_diversity': float(avg_cluster_diversity),
            'n_faces': n_faces
        }
    
    def create_novelty_visualizations(self, faces, similarity_matrix, clusters, novelty_metrics):
        """Create visualizations for novelty assessment"""
        print("📊 Creating novelty visualizations...")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Similarity matrix heatmap
        im1 = ax1.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Face Similarity Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Face Index', fontsize=12)
        ax1.set_ylabel('Face Index', fontsize=12)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Highlight clusters with rectangles
        for cluster_id, cluster_info in clusters.items():
            if cluster_info['size'] > 1:
                indices = cluster_info['indices']
                if len(indices) > 0:
                    min_idx = min(indices)
                    max_idx = max(indices)
                    rect = plt.Rectangle((min_idx-0.5, min_idx-0.5), 
                                        max_idx-min_idx+1, max_idx-min_idx+1,
                                        fill=False, edgecolor='red', linewidth=2)
                    ax1.add_patch(rect)
        
        # 2. Novelty metrics bar chart
        metrics = ['Avg Similarity', 'Max Similarity', 'Novelty %', 'Cluster Diversity']
        values = [
            novelty_metrics['avg_similarity'],
            novelty_metrics['max_similarity'],
            novelty_metrics['novelty_percentage'] / 100,  # Normalize to 0-1
            novelty_metrics['avg_cluster_diversity']
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Novelty Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value, metric in zip(bars, values, metrics):
            height = bar.get_height()
            if metric == 'Novelty %':
                label = f'{value*100:.1f}%'
            else:
                label = f'{value:.3f}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02, label,
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Cluster size distribution
        if clusters:
            cluster_sizes = [info['size'] for info in clusters.values()]
            cluster_ids = list(clusters.keys())
            
            # Only show first 20 clusters for clarity
            if len(cluster_sizes) > 20:
                cluster_sizes = cluster_sizes[:20]
                cluster_ids = cluster_ids[:20]
                ax3.set_xlabel('Cluster ID (first 20 shown)', fontsize=12)
            else:
                ax3.set_xlabel('Cluster ID', fontsize=12)
            
            x_pos = range(len(cluster_sizes))
            bars = ax3.bar(x_pos, cluster_sizes, alpha=0.8)
            ax3.set_ylabel('Number of Faces', fontsize=12)
            ax3.set_title(f'Cluster Size Distribution ({len(clusters)} clusters)', 
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add size labels
            for bar, size, x in zip(bars, cluster_sizes, x_pos):
                height = bar.get_height()
                ax3.text(x, height + 0.1, str(size), ha='center', va='bottom', fontsize=9)
            
            # Set x-ticks
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'C{idx}' for idx in cluster_ids], rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No clusters found\n(all faces are unique)',
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
            ax3.axis('off')
        
        # 4. Similarity distribution histogram
        mask = ~np.eye(len(faces), dtype=bool)
        similarities = similarity_matrix[mask]
        
        ax4.hist(similarities, bins=30, alpha=0.7, density=True, edgecolor='black')
        ax4.set_xlabel('Similarity Score', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Pairwise Similarity Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add vertical line for memorization threshold
        ax4.axvline(x=0.9, color='red', linestyle='--', label='High similarity (0.9)')
        ax4.legend()
        
        plt.suptitle('Novelty Assessment Analysis', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        novelty_viz_path = os.path.join(self.novelty_dir, "novelty_analysis_visualization.png")
        plt.savefig(novelty_viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Novelty analysis visualizations saved: {novelty_viz_path}")
        
        # Create example visualization of similar faces
        self.create_similar_faces_examples(faces, similarity_matrix, clusters)
        
        return novelty_viz_path
    
    def create_similar_faces_examples(self, faces, similarity_matrix, clusters):
        """Create visualization of similar faces (potential memorizations)"""
        # Find the most similar pair
        n_faces = len(faces)
        mask = ~np.eye(n_faces, dtype=bool)
        
        if similarity_matrix[mask].size > 0:
            # Find indices of maximum similarity (excluding diagonal)
            max_similarity = similarity_matrix[mask].max()
            
            if max_similarity > 0.8:  # Only show if actually similar
                # Find all pairs with this similarity
                indices = np.where((similarity_matrix == max_similarity) & mask)
                pairs = list(zip(indices[0], indices[1]))
                
                # Take first pair
                if pairs:
                    i, j = pairs[0]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    ax1.imshow(faces[i]['image_np'])
                    ax1.set_title(f'Face {i}\nSimilarity: {max_similarity:.3f}', fontsize=12)
                    ax1.axis('off')
                    
                    ax2.imshow(faces[j]['image_np'])
                    ax2.set_title(f'Face {j}', fontsize=12)
                    ax2.axis('off')
                    
                    plt.suptitle('Most Similar Face Pair (Potential Memorization)', 
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    similar_faces_path = os.path.join(self.novelty_dir, "most_similar_faces.png")
                    plt.savefig(similar_faces_path, dpi=150, bbox_inches='tight')
                    plt.show()
                    
                    print(f"✅ Similar faces example saved: {similar_faces_path}")
    
    def create_comprehensive_report(self, style_results, coverage_results, novelty_results):
        """Create comprehensive advanced diversity report"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ADVANCED DIVERSITY REPORT")
        print(f"{'='*60}")
        
        # Calculate overall scores
        style_score = style_results['style_diversity']['overall_style_score']
        
        # For coverage, use F1 score or average of precision/recall
        coverage_f1 = coverage_results['f1_score']
        coverage_avg = (coverage_results['precision'] + coverage_results['recall']) / 2
        coverage_score = max(coverage_f1, coverage_avg)  # Use the better score
        
        novelty_score = novelty_results['novelty_percentage'] / 100
        
        # Weighted overall score
        weights = {'style': 0.3, 'coverage': 0.4, 'novelty': 0.3}
        overall_score = (style_score * weights['style'] + 
                        coverage_score * weights['coverage'] + 
                        novelty_score * weights['novelty'])
        
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_type': 'Advanced Diversity Analysis',
            'n_faces_analyzed': novelty_results['total_faces'],
            'scores': {
                'style_diversity': float(style_score),
                'mode_coverage': float(coverage_score),
                'novelty': float(novelty_score),
                'overall': float(overall_score)
            },
            'detailed_results': {
                'style_analysis': style_results,
                'coverage_analysis': coverage_results,
                'novelty_analysis': novelty_results
            },
            'interpretation': self.interpret_advanced_diversity(overall_score),
            'recommendations': self.generate_advanced_recommendations(
                style_results, coverage_results, novelty_results, overall_score
            )
        }
        
        print(f"\n📋 ADVANCED DIVERSITY SCORES:")
        print(f"  Style Diversity: {style_score:.3f}")
        print(f"  Mode Coverage: {coverage_score:.3f}")
        print(f"  Novelty: {novelty_score:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
        
        print(f"\n📈 INTERPRETATION:")
        print(f"  {report['interpretation']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Save report
        report_path = os.path.join(self.output_dir, "comprehensive_advanced_diversity_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n Comprehensive report saved: {report_path}")
        
        # Create final visualization
        self.create_final_visualization(style_score, coverage_score, novelty_score, overall_score)
        
        return report
    
    def interpret_advanced_diversity(self, overall_score):
        """Interpret overall advanced diversity score"""
        if overall_score > 0.8:
            return "EXCELLENT - Highly diverse, well-covered, and novel generations"
        elif overall_score > 0.6:
            return "GOOD - Good diversity with some room for improvement"
        elif overall_score > 0.4:
            return "MODERATE - Some diversity issues present"
        else:
            return "LOW - Significant diversity, coverage, or novelty issues"
    
    def generate_advanced_recommendations(self, style_results, coverage_results, 
                                        novelty_results, overall_score):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Style recommendations
        style_div = style_results['style_diversity']
        if style_div.get('pose_diversity', 0) < 0.5:
            recommendations.append("Increase pose diversity in generated faces")
        
        if style_div.get('expression_diversity', 0) < 0.5:
            recommendations.append("Generate more varied facial expressions")
        
        if style_div.get('lighting_diversity', 0) < 0.5:
            recommendations.append("Increase lighting condition diversity")
        
        # Coverage recommendations
        if coverage_results.get('recall', 1) < 0.6:
            recommendations.append(f"Improve coverage of real distribution (recall: {coverage_results['recall']:.3f})")
        
        if coverage_results.get('precision', 1) < 0.8:
            recommendations.append("Maintain high precision while improving recall")
        
        # Novelty recommendations
        if novelty_results.get('novelty_percentage', 100) < 80:
            recommendations.append(f"Increase uniqueness (currently {novelty_results['novelty_percentage']:.1f}% unique)")
        
        if novelty_results.get('novelty_metrics', {}).get('max_similarity', 0) > 0.95:
            recommendations.append("Some faces are nearly identical - increase sampling diversity")
        
        # Overall recommendations based on your specific scores
        if coverage_results['recall'] < 0.5:
            recommendations.append("Focus on improving recall (distribution coverage) - try different random seeds")
        
        if overall_score < 0.7:
            recommendations.append("Consider adjusting guidance scale or sampling steps for better diversity")
        
        if len(recommendations) == 0:
            recommendations.append("Excellent diversity across all dimensions - maintain current approach")
        
        return recommendations
    
    def create_final_visualization(self, style_score, coverage_score, novelty_score, overall_score):
        """Create final visualization of all scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Scores bar chart
        categories = ['Style', 'Coverage', 'Novelty', 'Overall']
        scores = [style_score, coverage_score, novelty_score, overall_score]
        
        colors = ['skyblue', 'lightgreen', 'orange', 'purple']
        bars = ax1.bar(categories, scores, color=colors, alpha=0.8)
        
        ax1.set_ylabel('Score (0-1)', fontsize=12)
        ax1.set_title('Advanced Diversity Scores', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Overall score gauge
        ax2 = plt.subplot(122, polar=False)
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background
        ax2.plot(theta, r, color='gray', linewidth=10, alpha=0.3)
        
        # Score arc
        score_angle = overall_score * np.pi
        score_theta = np.linspace(0, score_angle, 100)
        score_r = np.ones_like(score_theta)
        
        # Color based on score
        if overall_score > 0.8:
            gauge_color = 'green'
        elif overall_score > 0.6:
            gauge_color = 'orange'
        else:
            gauge_color = 'red'
        
        ax2.plot(score_theta, score_r, color=gauge_color, linewidth=10)
        
        # Add needle
        needle_angle = overall_score * np.pi
        ax2.plot([needle_angle, needle_angle], [0.7, 1.1], color='black', linewidth=3)
        
        # Labels
        ax2.text(np.pi/2, 0.5, f'{overall_score:.3f}', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        
        if overall_score > 0.8:
            rating = "EXCELLENT"
        elif overall_score > 0.6:
            rating = "GOOD"
        elif overall_score > 0.4:
            rating = "MODERATE"
        else:
            rating = "LOW"
        
        ax2.text(np.pi/2, 0.3, rating, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax2.set_xlim([0, np.pi])
        ax2.set_ylim([0, 1.2])
        ax2.axis('off')
        ax2.set_title('Overall Advanced Diversity Score', fontsize=14, fontweight='bold')
        
        plt.suptitle('FFHQ Generated Faces - Advanced Diversity Assessment', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        final_viz_path = os.path.join(self.output_dir, "final_advanced_diversity_assessment.png")
        plt.savefig(final_viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Final visualization saved: {final_viz_path}")
    
    def run_complete_analysis(self):
        """Run complete advanced diversity analysis"""
        print(" RUNNING COMPLETE ADVANCED DIVERSITY ANALYSIS")
        print("=" * 60)
        
        # Load existing faces
        faces, source_dir = self.load_existing_faces()
        
        if faces is None or len(faces) < 10:
            print("❌ Need at least 10 faces for analysis!")
            return
        
        print(f"\n📊 Starting analysis on {len(faces)} faces...")
        
        # 1. Style Variation Analysis
        style_results = self.analyze_style_variation(faces)
        
        # 2. Mode Coverage Analysis
        coverage_results = self.analyze_mode_coverage(faces)
        
        # 3. Novelty Assessment
        novelty_results = self.analyze_novelty_assessment(faces)
        
        # 4. Comprehensive Report
        report = self.create_comprehensive_report(style_results, coverage_results, novelty_results)
        
        print(f"\n ADVANCED DIVERSITY ANALYSIS COMPLETE!")
        print(f" All results saved in: {self.output_dir}/")
        
        return report

def main():
    """Main function"""
    print(" FFHQ ADVANCED DIVERSITY ANALYSIS")
    print("=" * 60)
    print("\nThis script analyzes:")
    print("1. Style Variation (pose, lighting, expression)")
    print("2. Mode Coverage (precision-recall metrics)")
    print("3. Novelty Assessment (unique, non-memorized generations)")
    
    # Check for existing directories
    import glob
    existing_dirs = sorted(glob.glob("diversity_analysis_*"))
    
    if existing_dirs:
        latest_dir = existing_dirs[-1]
        print(f"\n Found existing analysis: {latest_dir}")
        use_existing = input(f"Use existing faces from {latest_dir}? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            analyzer = AdvancedDiversityAnalyzer(latest_dir)
        else:
            analyzer = AdvancedDiversityAnalyzer()
    else:
        analyzer = AdvancedDiversityAnalyzer()
    
    print(f"\nOptions:")
    print("1. Run complete advanced analysis")
    print("2. Analyze style variation only")
    print("3. Analyze mode coverage only")
    print("4. Analyze novelty only")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            analyzer.run_complete_analysis()
            break
        elif choice == "2":
            faces, _ = analyzer.load_existing_faces()
            if faces:
                analyzer.analyze_style_variation(faces)
            break
        elif choice == "3":
            faces, _ = analyzer.load_existing_faces()
            if faces:
                analyzer.analyze_mode_coverage(faces)
            break
        elif choice == "4":
            faces, _ = analyzer.load_existing_faces()
            if faces:
                analyzer.analyze_novelty_assessment(faces)
            break
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
