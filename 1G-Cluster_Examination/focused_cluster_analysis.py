#focused_cluster_analysis.py
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from datetime import datetime
from tqdm import tqdm
import colorsys
from scipy.ndimage import sobel, gaussian_laplace, gaussian_filter
from PIL import ImageDraw
warnings.filterwarnings('ignore')

class FocusedClusterAnalyzer:
    def __init__(self):
        """Initialize focused cluster analyzer for gender/age/ethnicity"""
        print(" FOCUSED CLUSTER ANALYSIS (Gender/Age/Ethnicity)")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"focused_cluster_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.clusters_dir = os.path.join(self.output_dir, "clusters")
        
        for dir_path in [self.visualizations_dir, self.clusters_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f" Output directory: {self.output_dir}")
    
    def load_images(self, images_dir):
        """Load images from directory"""
        print(f"\n Loading images from: {images_dir}")
        
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(png_files) == 0:
            print("❌ No PNG files found.")
            return None, None
        
        print(f" Found {len(png_files)} images")
        
        # Load images
        images = []
        image_info = []
        
        for idx, filename in enumerate(tqdm(png_files, desc="Loading images")):
            img_path = os.path.join(images_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img)
            
            images.append(img_np)
            image_info.append({
                'id': idx,
                'path': img_path,
                'filename': filename,
                'image': img
            })
        
        return np.array(images), image_info
    
    def extract_face_features(self, images):
        """Extract features relevant to face analysis"""
        print("\n Extracting face-relevant features...")
        
        features = []
        
        for img in tqdm(images, desc="Extracting features"):
            img_features = []
            
            # Convert to float and normalize
            img_float = img.astype(np.float32) / 255.0
            
            # 1. Skin tone detection (simplified)
            # Assuming faces are in the center of FFHQ images
            height, width = img_float.shape[:2]
            center_h, center_w = height // 2, width // 2
            crop_size = min(height, width) // 3
            
            # Extract center region (where face usually is)
            top = max(0, center_h - crop_size)
            bottom = min(height, center_h + crop_size)
            left = max(0, center_w - crop_size)
            right = min(width, center_w + crop_size)
            
            face_region = img_float[top:bottom, left:right]
            
            if len(face_region) > 0:
                # Average skin color (RGB)
                avg_color = np.mean(face_region.reshape(-1, 3), axis=0)
                img_features.extend(avg_color)
                
                # Skin tone brightness
                brightness = np.mean(avg_color)
                img_features.append(brightness)
                
                # Skin tone variation
                color_std = np.std(face_region.reshape(-1, 3), axis=0)
                img_features.extend(color_std)
                img_features.append(np.mean(color_std))
            
            # 2. Hair color detection (top region)
            hair_region = img_float[:height//4, :]
            if len(hair_region) > 0:
                hair_color = np.mean(hair_region.reshape(-1, 3), axis=0)
                img_features.extend(hair_color)
                hair_brightness = np.mean(hair_color)
                img_features.append(hair_brightness)
            else:
                img_features.extend([0, 0, 0, 0])
            
            # 3. Face shape features (simplified edge detection)
            if len(img_float.shape) == 3:
                gray = np.mean(img_float, axis=2)
            else:
                gray = img_float
            
            # Edge density in center (face outline)
            center_region = gray[top:bottom, left:right]
            if len(center_region) > 0:
                grad_x = sobel(center_region, axis=0)
                grad_y = sobel(center_region, axis=1)
                edge_strength = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                img_features.append(edge_strength)
            else:
                img_features.append(0)
            
            # 4. Contrast features (for age - wrinkles/texture)
            contrast = np.std(gray)
            img_features.append(contrast)
            
            # 5. Symmetry (simplified)
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            # Flip right half for comparison
            right_half_flipped = np.fliplr(right_half)
            
            # Ensure same shape
            min_height = min(left_half.shape[0], right_half_flipped.shape[0])
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            
            if min_height > 0 and min_width > 0:
                left_crop = left_half[:min_height, :min_width]
                right_crop = right_half_flipped[:min_height, :min_width]
                symmetry_score_val = np.mean(np.abs(left_crop - right_crop))
                img_features.append(symmetry_score_val)
            else:
                img_features.append(0)
            
            features.append(img_features)
        
        feature_matrix = np.array(features)
        print(f" Extracted {feature_matrix.shape[1]} face-relevant features")
        
        return feature_matrix
    
    def analyze_gender_patterns(self, images, labels, image_info):
        """Analyze potential gender patterns in clusters"""
        print("\n Analyzing potential gender patterns...")
        
        unique_labels = np.unique(labels)
        gender_analysis = {}
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            
            # Analyze facial features that might indicate gender
            features = []
            for img in cluster_images:
                img_float = img.astype(np.float32) / 255.0
                
                # Jawline width estimation (simplified)
                height, width = img_float.shape[:2]
                bottom_region = img_float[3*height//4:, :]
                
                if len(bottom_region.shape) == 3:
                    bottom_gray = np.mean(bottom_region, axis=2)
                else:
                    bottom_gray = bottom_region
                
                # Edge detection in bottom region (jawline)
                edges = np.abs(sobel(bottom_gray))
                jaw_strength = np.mean(edges)
                
                # Brow ridge prominence (top-center intensity gradient)
                brow_region = img_float[height//3:height//2, width//3:2*width//3]
                if len(brow_region) > 0:
                    brow_contrast = np.std(brow_region)
                else:
                    brow_contrast = 0
                
                features.append([jaw_strength, brow_contrast])
            
            if len(features) > 0:
                avg_features = np.mean(features, axis=0)
                
                # Simple gender estimation based on features
                jaw_strength, brow_contrast = avg_features
                
                # These thresholds are arbitrary and would need tuning
                if jaw_strength > 0.05 and brow_contrast > 0.1:
                    gender_guess = "Potentially male"
                elif jaw_strength < 0.03 and brow_contrast < 0.08:
                    gender_guess = "Potentially female"
                else:
                    gender_guess = "Unclear"
                
                gender_analysis[int(cluster_id)] = {
                    'size': len(cluster_images),
                    'avg_jaw_strength': float(jaw_strength),
                    'avg_brow_contrast': float(brow_contrast),
                    'gender_estimation': gender_guess,
                    'sample_images': cluster_indices[:5].tolist()
                }
                
                print(f"   Cluster {cluster_id}: {gender_guess}")
                print(f"      Size: {len(cluster_images)} images")
                print(f"      Jaw strength: {jaw_strength:.3f}")
                print(f"      Brow contrast: {brow_contrast:.3f}")
        
        return gender_analysis
    
    def analyze_age_patterns(self, images, labels, image_info):
        """Analyze potential age patterns in clusters"""
        print("\n Analyzing potential age patterns...")
        
        unique_labels = np.unique(labels)
        age_analysis = {}
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            
            # Analyze features that might indicate age
            features = []
            for img in cluster_images:
                img_float = img.astype(np.float32) / 255.0
                
                if len(img_float.shape) == 3:
                    gray = np.mean(img_float, axis=2)
                else:
                    gray = img_float
                
                # Wrinkle/texture analysis (high frequency content)
                try:
                    laplacian = gaussian_laplace(gray, sigma=1)
                    texture_energy = np.mean(np.abs(laplacian))
                except:
                    texture_energy = 0
                
                # Skin smoothness (low frequency dominance)
                smoothed = gaussian_filter(gray, sigma=2)
                smoothness = np.mean(np.abs(gray - smoothed))
                
                # Overall contrast (older faces often have more contrast)
                contrast = np.std(gray)
                
                features.append([texture_energy, smoothness, contrast])
            
            if len(features) > 0:
                avg_features = np.mean(features, axis=0)
                texture_energy, smoothness, contrast = avg_features
                
                # Simple age estimation
                if texture_energy > 0.02 and contrast > 0.15:
                    age_guess = "Potentially older"
                elif texture_energy < 0.01 and smoothness > 0.95:
                    age_guess = "Potentially younger"
                else:
                    age_guess = "Middle age or unclear"
                
                age_analysis[int(cluster_id)] = {
                    'size': len(cluster_images),
                    'texture_energy': float(texture_energy),
                    'smoothness': float(smoothness),
                    'contrast': float(contrast),
                    'age_estimation': age_guess,
                    'sample_images': cluster_indices[:5].tolist()
                }
                
                print(f"   Cluster {cluster_id}: {age_guess}")
                print(f"      Texture: {texture_energy:.3f}, Smoothness: {smoothness:.3f}")
        
        return age_analysis
    
    def analyze_ethnicity_patterns(self, images, labels, image_info):
        """Analyze potential ethnicity patterns based on skin tone"""
        print("\n Analyzing potential ethnicity patterns...")
        
        unique_labels = np.unique(labels)
        ethnicity_analysis = {}
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            
            # Analyze skin tone colors
            skin_tones = []
            for img in cluster_images:
                img_float = img.astype(np.float32) / 255.0
                
                # Extract center region (face)
                height, width = img_float.shape[:2]
                center_h, center_w = height // 2, width // 2
                crop_size = min(height, width) // 4
                
                top = max(0, center_h - crop_size)
                bottom = min(height, center_h + crop_size)
                left = max(0, center_w - crop_size)
                right = min(width, center_w + crop_size)
                
                face_region = img_float[top:bottom, left:right]
                
                if len(face_region) > 0 and len(face_region.shape) == 3:
                    avg_skin_color = np.mean(face_region.reshape(-1, 3), axis=0)
                    skin_tones.append(avg_skin_color)
            
            if len(skin_tones) > 0:
                avg_skin_tone = np.mean(skin_tones, axis=0)
                std_skin_tone = np.std(skin_tones, axis=0)
                
                # Convert to HSV for better skin tone analysis
                avg_rgb = avg_skin_tone
                avg_hsv = colorsys.rgb_to_hsv(avg_rgb[0], avg_rgb[1], avg_rgb[2])
                
                # Simple ethnicity estimation based on skin tone
                hue, saturation, value = avg_hsv
                
                if value > 0.7 and saturation < 0.3:
                    ethnicity_guess = "Potentially lighter skin tone"
                elif value < 0.4:
                    ethnicity_guess = "Potentially darker skin tone"
                elif saturation > 0.5 and hue > 0.05 and hue < 0.15:
                    ethnicity_guess = "Potentially medium/warm skin tone"
                else:
                    ethnicity_guess = "Mixed or unclear"
                
                ethnicity_analysis[int(cluster_id)] = {
                    'size': len(cluster_images),
                    'avg_skin_r': float(avg_skin_tone[0]),
                    'avg_skin_g': float(avg_skin_tone[1]),
                    'avg_skin_b': float(avg_skin_tone[2]),
                    'skin_brightness': float(value),
                    'skin_saturation': float(saturation),
                    'ethnicity_estimation': ethnicity_guess,
                    'sample_images': cluster_indices[:5].tolist()
                }
                
                print(f"   Cluster {cluster_id}: {ethnicity_guess}")
                print(f"      RGB: ({avg_skin_tone[0]:.2f}, {avg_skin_tone[1]:.2f}, {avg_skin_tone[2]:.2f})")
                print(f"      Brightness: {value:.2f}, Saturation: {saturation:.2f}")
        
        return ethnicity_analysis
    
    def create_cluster_visualizations(self, images, labels, image_info, 
                                     gender_analysis, age_analysis, ethnicity_analysis):
        """Create comprehensive visualizations of cluster analysis"""
        print("\n Creating cluster analysis visualizations...")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # 1. Create feature space visualization
        plt.figure(figsize=(15, 10))
        
        # Extract features for visualization
        features = []
        for img in images:
            img_float = img.astype(np.float32) / 255.0
            
            if len(img_float.shape) == 3:
                gray = np.mean(img_float, axis=2)
            else:
                gray = img_float
            
            # Simple features for visualization
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.append([brightness, contrast])
        
        features = np.array(features)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7, s=50)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Cluster Visualization ({n_clusters} clusters)')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # 2. Cluster sizes
        plt.subplot(2, 2, 2)
        cluster_sizes = [np.sum(labels == i) for i in unique_labels]
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        bars = plt.bar([f'Cluster {i}' for i in unique_labels], cluster_sizes, color=colors)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Images')
        plt.title('Cluster Sizes')
        
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom')
        
        # 3. Gender analysis visualization
        plt.subplot(2, 2, 3)
        if gender_analysis:
            jaw_strengths = [gender_analysis[int(i)]['avg_jaw_strength'] for i in unique_labels]
            brow_contrasts = [gender_analysis[int(i)]['avg_brow_contrast'] for i in unique_labels]
            
            scatter = plt.scatter(jaw_strengths, brow_contrasts, 
                                c=unique_labels, cmap='tab10', s=100)
            
            # Add gender regions (simplified)
            plt.axvline(x=0.04, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0.09, color='gray', linestyle='--', alpha=0.5)
            
            plt.text(0.02, 0.12, 'Potentially\nfemale', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))
            plt.text(0.06, 0.06, 'Potentially\nmale', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.xlabel('Average Jaw Strength')
            plt.ylabel('Average Brow Contrast')
            plt.title('Gender Pattern Analysis')
            plt.grid(True, alpha=0.3)
        
        # 4. Age analysis visualization
        plt.subplot(2, 2, 4)
        if age_analysis:
            texture_energies = [age_analysis[int(i)]['texture_energy'] for i in unique_labels]
            contrasts = [age_analysis[int(i)]['contrast'] for i in unique_labels]
            
            scatter = plt.scatter(texture_energies, contrasts, 
                                c=unique_labels, cmap='tab10', s=100)
            
            # Add age regions
            plt.axvline(x=0.015, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0.125, color='gray', linestyle='--', alpha=0.5)
            
            plt.text(0.005, 0.10, 'Potentially\nyounger', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            plt.text(0.02, 0.15, 'Potentially\nolder', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
            
            plt.xlabel('Texture Energy (wrinkles)')
            plt.ylabel('Contrast')
            plt.title('Age Pattern Analysis')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('Focused Cluster Analysis - Gender and Age Patterns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "cluster_demographic_analysis.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create cluster collages
        self.create_cluster_collages(images, labels, image_info)
    
    def create_cluster_collages(self, images, labels, image_info):
        """Create collages for each cluster"""
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_images = [images[i] for i in cluster_indices]
            
            if len(cluster_images) == 0:
                continue
            
            # Create collage
            n_images = len(cluster_images)
            grid_size = int(np.ceil(np.sqrt(n_images)))
            thumb_size = 120
            
            collage_width = grid_size * thumb_size
            collage_height = grid_size * thumb_size
            collage = Image.new('RGB', (collage_width, collage_height), color='white')
            
            for idx, img_array in enumerate(cluster_images[:grid_size**2]):  # Limit to grid size
                row = idx // grid_size
                col = idx % grid_size
                
                img = Image.fromarray(img_array)
                img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                
                x = col * thumb_size
                y = row * thumb_size
                
                collage.paste(img, (x, y))
                
                # Add sample number
                draw = ImageDraw.Draw(collage)
                sample_num = cluster_indices[idx]
                draw.text((x + 5, y + 5), str(sample_num), 
                         fill='white', stroke_width=2, stroke_fill='black')
            
            # Save collage
            collage.save(os.path.join(self.clusters_dir, f"cluster_{cluster_id}_collage.png"))
            
            print(f"   Saved collage for Cluster {cluster_id} ({len(cluster_images)} images)")
    
    def generate_report(self, images, labels, image_info, 
                       gender_analysis, age_analysis, ethnicity_analysis):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*60}")
        print("GENERATING ANALYSIS REPORT")
        print(f"{'='*60}")
        
        n_images = len(images)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        report = f"""
FOCUSED CLUSTER ANALYSIS REPORT
===============================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.output_dir}

1. DATASET SUMMARY
-----------------
• Total images analyzed: {n_images}
• Number of clusters found: {n_clusters}

2. CLUSTER DISTRIBUTION
----------------------
"""
        
        for cluster_id in unique_labels:
            cluster_size = np.sum(labels == cluster_id)
            percentage = (cluster_size / n_images) * 100
            report += f"• Cluster {cluster_id}: {cluster_size} images ({percentage:.1f}%)\n"
        
        report += f"""
3. GENDER ANALYSIS
-----------------
"""
        
        if gender_analysis:
            for cluster_id, analysis in gender_analysis.items():
                report += f"""
Cluster {cluster_id}:
• Size: {analysis['size']} images
• Jaw strength: {analysis['avg_jaw_strength']:.3f}
• Brow contrast: {analysis['avg_brow_contrast']:.3f}
• Gender estimation: {analysis['gender_estimation']}
• Sample images: {analysis['sample_images']}
"""
        
        report += f"""
4. AGE ANALYSIS
--------------
"""
        
        if age_analysis:
            for cluster_id, analysis in age_analysis.items():
                report += f"""
Cluster {cluster_id}:
• Texture energy: {analysis['texture_energy']:.3f} (higher = more wrinkles)
• Smoothness: {analysis['smoothness']:.3f} (higher = smoother skin)
• Contrast: {analysis['contrast']:.3f} (higher = more age-related contrast)
• Age estimation: {analysis['age_estimation']}
"""
        
        report += f"""
5. ETHNICITY ANALYSIS
--------------------
"""
        
        if ethnicity_analysis:
            for cluster_id, analysis in ethnicity_analysis.items():
                report += f"""
Cluster {cluster_id}:
• Average skin tone (RGB): ({analysis['avg_skin_r']:.2f}, {analysis['avg_skin_g']:.2f}, {analysis['avg_skin_b']:.2f})
• Brightness: {analysis['skin_brightness']:.2f}
• Saturation: {analysis['skin_saturation']:.2f}
• Ethnicity estimation: {analysis['ethnicity_estimation']}
"""
        
        report += f"""
6. KEY FINDINGS
--------------
• Check cluster collages in: {self.clusters_dir}/
• Examine visual patterns in each cluster

========================================================================
"""
        
        # Save report
        report_file = os.path.join(self.output_dir, "focused_cluster_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_file}")
        
        # Save analysis data
        analysis_data = {
            'n_images': int(n_images),
            'n_clusters': int(n_clusters),
            'labels': labels.tolist(),
            'gender_analysis': gender_analysis,
            'age_analysis': age_analysis,
            'ethnicity_analysis': ethnicity_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "analysis_data.json"), 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"✅ Analysis data saved to: {os.path.join(self.output_dir, 'analysis_data.json')}")
        
        return report
    
    def run_analysis(self):
        """Run focused cluster analysis"""
        print(f"\n{'='*60}")
        print("RUNNING FOCUSED CLUSTER ANALYSIS")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Load images
        images_dir = "latent_analysis_20251202_175108/latents/"
        
        if not os.path.exists(images_dir):
            possible_paths = [
                "latent_analysis_20251202_175108/latents/",
                "./latent_analysis_20251202_175108/latents/",
                "../latent_analysis_20251202_175108/latents/",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    images_dir = path
                    print(f" Using images directory: {images_dir}")
                    break
        
        if not os.path.exists(images_dir):
            print("❌ No images directory found.")
            print("💡 Please specify the correct path to your latents directory.")
            return None
        
        images, image_info = self.load_images(images_dir)
        if images is None:
            return None
        
        print(f"\n✅ Starting analysis with {len(images)} images")
        
        # Extract face-relevant features
        feature_matrix = self.extract_face_features(images)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Perform clustering (try 2-4 clusters for face analysis)
        print("\n Performing clustering...")
        
        # Try different cluster counts
        best_silhouette = -1
        best_labels = None
        best_n_clusters = 2
        
        for n_clusters in [2, 3, 4]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                print(f"   {n_clusters} clusters: silhouette = {silhouette:.3f}")
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_labels = labels
                    best_n_clusters = n_clusters
        
        print(f"⭐ Selected {best_n_clusters} clusters (silhouette: {best_silhouette:.3f})")
        
        # Analyze patterns
        gender_analysis = self.analyze_gender_patterns(images, best_labels, image_info)
        age_analysis = self.analyze_age_patterns(images, best_labels, image_info)
        ethnicity_analysis = self.analyze_ethnicity_patterns(images, best_labels, image_info)
        
        # Create visualizations
        self.create_cluster_visualizations(images, best_labels, image_info,
                                         gender_analysis, age_analysis, ethnicity_analysis)
        
        # Generate report
        report = self.generate_report(images, best_labels, image_info,
                                     gender_analysis, age_analysis, ethnicity_analysis)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Total time: {elapsed_time.total_seconds():.1f} seconds")
        print(f"✅ Output saved to: {self.output_dir}/")
        print(f"✅ Report: {self.output_dir}/focused_cluster_analysis_report.txt")
        print(f"✅ Visualizations: {self.visualizations_dir}/")
        print(f"✅ Cluster collages: {self.clusters_dir}/")
        print(f"\n Analysis Summary:")
        print(f"   • Analyzed {len(images)} face images")
        print(f"   • Found {best_n_clusters} distinct clusters")
        print(f"   • Generated gender, age, and ethnicity estimations")
        print(f"   • Created detailed visualizations and collages")
        print(f"\n📋 Next steps:")
        print(f"   1. Review the cluster collages")
        print(f"   2. Examine the demographic analysis plots")
        print(f"   3. Validate algorithmic estimates with human perception")
        print(f"   4. Use findings for dataset curation or bias analysis")
        
        return {
            'images': images,
            'labels': best_labels,
            'image_info': image_info,
            'gender_analysis': gender_analysis,
            'age_analysis': age_analysis,
            'ethnicity_analysis': ethnicity_analysis,
            'output_dir': self.output_dir
        }


# Main execution
if __name__ == "__main__":
    analyzer = FocusedClusterAnalyzer()
    results = analyzer.run_analysis()
