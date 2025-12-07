# ffhq_analyze_existing_faces.py
import torch
import os
import sys
import numpy as np
from PIL import Image
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import glob

class FaceAnalyzer:
    def __init__(self, faces_dir=None):
        print("📊 EXISTING FACE ANALYZER")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"face_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📁 Analysis output: {self.output_dir}")
        
        # Set faces directory
        if faces_dir and os.path.exists(faces_dir):
            self.faces_dir = faces_dir
        else:
            # Find the most recent faces directory
            face_dirs = sorted(glob.glob("diverse_faces_*"))
            if face_dirs:
                self.faces_dir = os.path.join(face_dirs[-1], "faces")
            else:
                print("❌ No faces directory found!")
                print("   Provide path with --faces_dir")
                sys.exit(1)
        
        print(f"📂 Analyzing faces from: {self.faces_dir}")
    
    def load_existing_faces(self):
        """Load existing generated faces"""
        print(f"\n📥 LOADING EXISTING FACES")
        print("=" * 40)
        
        # Find all PNG files
        image_files = glob.glob(os.path.join(self.faces_dir, "*.png"))
        
        if not image_files:
            print(f"❌ No PNG files found in {self.faces_dir}")
            return []
        
        print(f"Found {len(image_files)} face images")
        
        faces = []
        
        for i, img_file in enumerate(image_files):
            try:
                # Load image
                img = Image.open(img_file).convert('RGB')
                img_np = np.array(img) / 255.0  # Normalize to 0-1
                
                # Extract seed from filename if possible
                filename = os.path.basename(img_file)
                seed = self.extract_seed_from_filename(filename)
                
                face_data = {
                    'id': i,
                    'filename': filename,
                    'filepath': img_file,
                    'image_np': img_np,
                    'image': img,
                    'seed': seed,
                    'width': img_np.shape[1],
                    'height': img_np.shape[0]
                }
                
                faces.append(face_data)
                
                print(f"  Loaded: {filename} ({img_np.shape[1]}x{img_np.shape[0]})")
                
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
        
        print(f"✅ Successfully loaded {len(faces)} faces")
        
        # Save loading summary
        self.save_loading_summary(faces)
        
        return faces
    
    def extract_seed_from_filename(self, filename):
        """Extract seed number from filename"""
        import re
        match = re.search(r'seed(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def save_loading_summary(self, faces):
        """Save summary of loaded faces"""
        summary = {
            'faces_dir': self.faces_dir,
            'total_faces': len(faces),
            'image_shapes': [list(f['image_np'].shape) for f in faces],
            'filenames': [f['filename'] for f in faces],
            'seeds': [f['seed'] for f in faces],
            'loaded_at': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_dir, "loading_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Loading summary saved: {summary_path}")
        
        return summary_path
    
    def analyze_novelty(self, faces):
        """Analyze novelty of existing faces"""
        print(f"\n📊 ANALYZING NOVELTY")
        print("=" * 40)
        
        if len(faces) < 2:
            print("❌ Need at least 2 faces for novelty analysis")
            return None
        
        print(f"Analyzing {len(faces)} faces...")
        
        # Extract features
        features = []
        for face in faces:
            img_np = face['image_np']
            
            # Simple features for diversity analysis
            color_features = img_np.mean(axis=(0, 1)).flatten()  # RGB means
            brightness = img_np.mean()  # Overall brightness
            contrast = img_np.std()  # Overall contrast
            
            # Texture/edge features
            gray = np.mean(img_np, axis=2)
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.hypot(sobel_x, sobel_y)
            edge_mean = edge_magnitude.mean()
            
            # Combine all features
            feature_vector = np.concatenate([
                color_features,
                np.array([brightness, contrast, edge_mean])
            ])
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Calculate similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(features)
        np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1.0
        
        # Calculate novelty metrics
        n_faces = len(faces)
        mask = ~np.eye(n_faces, dtype=bool)  # Mask to exclude diagonal
        
        avg_similarity = similarity_matrix[mask].mean()
        max_similarity = similarity_matrix[mask].max()
        
        # Count unique faces (similarity < 0.9 with any other face)
        n_unique = 0
        for i in range(n_faces):
            is_unique = True
            for j in range(n_faces):
                if i != j and similarity_matrix[i, j] > 0.9:  # 90% similarity threshold
                    is_unique = False
                    break
            if is_unique:
                n_unique += 1
        
        novelty_percentage = (n_unique / n_faces) * 100
        
        print(f"\n📈 NOVELTY RESULTS:")
        print(f"  Total faces analyzed: {n_faces}")
        print(f"  Unique faces: {n_unique}")
        print(f"  Novelty percentage: {novelty_percentage:.1f}%")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Maximum similarity: {max_similarity:.3f}")
        print(f"  Minimum similarity: {similarity_matrix[mask].min():.3f}")
        
        # Show similarity matrix
        print(f"\n🔢 Pairwise similarities:")
        for i in range(min(5, n_faces)):  # Show first 5
            for j in range(min(5, n_faces)):
                if i != j:
                    print(f"  Face {i} vs Face {j}: {similarity_matrix[i, j]:.3f}")
        
        # Create visualization
        viz_path = self.create_novelty_visualization(faces, similarity_matrix, 
                                                    novelty_percentage, avg_similarity)
        
        # Save results
        results = {
            'novelty_percentage': float(novelty_percentage),
            'avg_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'min_similarity': float(similarity_matrix[mask].min()),
            'n_faces': n_faces,
            'n_unique': n_unique,
            'seeds': [f['seed'] for f in faces],
            'similarity_matrix': similarity_matrix.tolist()
        }
        
        results_path = os.path.join(self.output_dir, "novelty_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved: {results_path}")
        
        return results
    
    def create_novelty_visualization(self, faces, similarity_matrix, 
                                   novelty_percentage, avg_similarity):
        """Create visualization of novelty results"""
        print("📊 Creating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        n_faces = len(faces)
        
        # 1. Similarity matrix heatmap
        im = ax1.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Face Similarity Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Face Index', fontsize=10)
        ax1.set_ylabel('Face Index', fontsize=10)
        ax1.set_xticks(range(n_faces))
        ax1.set_yticks(range(n_faces))
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # Add similarity values to matrix
        for i in range(n_faces):
            for j in range(n_faces):
                if i != j:
                    ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8, 
                            color='white' if similarity_matrix[i, j] > 0.5 else 'black')
        
        # 2. Novelty gauge
        ax2 = plt.subplot(222, projection='polar')
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Score arc
        score = novelty_percentage / 100
        score_angle = score * np.pi
        score_theta = np.linspace(0, score_angle, 100)
        score_r = np.ones_like(score_theta)
        
        # Color based on score
        if score > 0.7:
            gauge_color = 'green'
        elif score > 0.5:
            gauge_color = 'orange'
        else:
            gauge_color = 'red'
        
        ax2.plot(score_theta, score_r, color=gauge_color, linewidth=8)
        ax2.plot(theta, r, color='gray', linewidth=8, alpha=0.3)
        
        # Add needle
        ax2.plot([score_angle, score_angle], [0.7, 1.1], color='black', linewidth=2)
        
        # Labels
        ax2.text(np.pi/2, 0.5, f'{novelty_percentage:.1f}%', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        if score > 0.7:
            rating = "EXCELLENT"
        elif score > 0.5:
            rating = "GOOD"
        elif score > 0.3:
            rating = "MODERATE"
        else:
            rating = "LOW"
        
        ax2.text(np.pi/2, 0.3, rating, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax2.set_xlim([0, np.pi])
        ax2.set_ylim([0, 1.2])
        ax2.axis('off')
        ax2.set_title('Novelty Score', fontsize=12, fontweight='bold')
        
        # 3. Similarity distribution
        mask = ~np.eye(n_faces, dtype=bool)
        similarities = similarity_matrix[mask]
        
        ax3.hist(similarities, bins=min(10, len(similarities)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=avg_similarity, color='red', linestyle='--', 
                   linewidth=2, label=f'Average: {avg_similarity:.3f}')
        ax3.axvline(x=0.9, color='orange', linestyle=':', 
                   linewidth=2, label='Threshold: 0.9')
        
        ax3.set_xlabel('Similarity Score', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title(f'Similarity Distribution ({len(similarities)} pairs)', 
                     fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Show sample faces (first 4)
        ax4.axis('off')
        if n_faces >= 4:
            # Create subgrid for faces
            gs = ax4.get_gridspec()
            ax4.remove()
            face_grid = fig.add_gridspec(2, 2, left=0.65, right=0.95, 
                                        bottom=0.05, top=0.45)
            
            for idx in range(min(4, n_faces)):
                row = idx // 2
                col = idx % 2
                ax_face = fig.add_subplot(face_grid[row, col])
                ax_face.imshow(faces[idx]['image_np'])
                ax_face.set_title(f"Face {idx} (seed: {faces[idx]['seed']})", 
                                fontsize=9)
                ax_face.axis('off')
        else:
            ax4.text(0.5, 0.5, f'Showing {n_faces} face(s)', 
                    ha='center', va='center', fontsize=12)
            for idx, face in enumerate(faces):
                ax4.imshow(face['image_np'])
                ax4.set_title(f"Face {idx}")
                ax4.axis('off')
        
        plt.suptitle(f'Novelty Analysis: {novelty_percentage:.1f}% Unique Faces', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        viz_path = os.path.join(self.output_dir, "novelty_analysis.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved: {viz_path}")
        
        return viz_path
    
    def check_week1_targets(self, novelty_results):
        """Check if we meet Week 1 targets"""
        print(f"\n🎯 WEEK 1 TARGET CHECK")
        print("=" * 40)
        
        if novelty_results is None:
            print("❌ No novelty results to check")
            return False
        
        novelty_pct = novelty_results['novelty_percentage']
        avg_sim = novelty_results['avg_similarity']
        
        print(f"Target 1: Novelty > 50%")
        print(f"  Your score: {novelty_pct:.1f}%")
        print(f"  Status: {'✅ PASS' if novelty_pct > 50 else '❌ FAIL'}")
        
        print(f"\nTarget 2: Average similarity < 0.8")
        print(f"  Your score: {avg_sim:.3f}")
        print(f"  Status: {'✅ PASS' if avg_sim < 0.8 else '❌ FAIL'}")
        
        print(f"\nTarget 3: Both targets met")
        both_met = novelty_pct > 50 and avg_sim < 0.8
        print(f"  Status: {'✅ ALL TARGETS MET!' if both_met else '❌ TARGETS NOT MET'}")
        
        # Interpretation
        if novelty_pct > 90:
            interpretation = "EXCELLENT novelty (highly unique)"
        elif novelty_pct > 70:
            interpretation = "GOOD novelty"
        elif novelty_pct > 50:
            interpretation = "MODERATE novelty (meets target)"
        elif novelty_pct > 30:
            interpretation = "LOW novelty (below target)"
        else:
            interpretation = "VERY LOW novelty (significant repetition)"
        
        print(f"\n📋 INTERPRETATION: {interpretation}")
        
        # Save target check
        target_check = {
            'week1_targets': {
                'novelty_gt_50': novelty_pct > 50,
                'avg_similarity_lt_08': avg_sim < 0.8,
                'both_met': both_met
            },
            'your_scores': {
                'novelty_percentage': novelty_pct,
                'avg_similarity': avg_sim
            },
            'interpretation': interpretation,
            'check_date': datetime.now().isoformat()
        }
        
        target_path = os.path.join(self.output_dir, "target_check.json")
        with open(target_path, 'w') as f:
            json.dump(target_check, f, indent=2)
        
        print(f"\n📄 Target check saved: {target_path}")
        
        return both_met
    
    def analyze_style_diversity(self, faces):
        """Analyze style diversity: brightness, contrast, color"""
        print(f"\n🎨 ANALYZING STYLE DIVERSITY")
        print("=" * 40)
        
        if len(faces) < 2:
            print("❌ Need at least 2 faces")
            return None
        
        style_metrics = []
        
        for face in faces:
            img_np = face['image_np']
            
            # Brightness
            brightness = img_np.mean()
            
            # Contrast
            contrast = img_np.std()
            
            # Color balance (RGB ratios)
            r, g, b = img_np.mean(axis=(0, 1))
            color_balance = [r, g, b]
            
            # Color temperature (simplified)
            color_temp = r / (g + 1e-8)  # Red/green ratio
            
            style_metrics.append({
                'brightness': float(brightness),
                'contrast': float(contrast),
                'color_r': float(r),
                'color_g': float(g),
                'color_b': float(b),
                'color_temp': float(color_temp)
            })
        
        # Calculate variation
        brightness_std = np.std([m['brightness'] for m in style_metrics])
        contrast_std = np.std([m['contrast'] for m in style_metrics])
        
        print(f"📊 STYLE DIVERSITY:")
        print(f"  Brightness variation: {brightness_std:.3f}")
        print(f"  Contrast variation: {contrast_std:.3f}")
        print(f"  Color range:")
        print(f"    Red: {np.min([m['color_r'] for m in style_metrics]):.3f} - {np.max([m['color_r'] for m in style_metrics]):.3f}")
        print(f"    Green: {np.min([m['color_g'] for m in style_metrics]):.3f} - {np.max([m['color_g'] for m in style_metrics]):.3f}")
        print(f"    Blue: {np.min([m['color_b'] for m in style_metrics]):.3f} - {np.max([m['color_b'] for m in style_metrics]):.3f}")
        
        # Save style analysis
        style_results = {
            'style_metrics': style_metrics,
            'variation': {
                'brightness_std': float(brightness_std),
                'contrast_std': float(contrast_std)
            },
            'color_ranges': {
                'red': [float(np.min([m['color_r'] for m in style_metrics])),
                       float(np.max([m['color_r'] for m in style_metrics]))],
                'green': [float(np.min([m['color_g'] for m in style_metrics])),
                         float(np.max([m['color_g'] for m in style_metrics]))],
                'blue': [float(np.min([m['color_b'] for m in style_metrics])),
                        float(np.max([m['color_b'] for m in style_metrics]))]
            }
        }
        
        style_path = os.path.join(self.output_dir, "style_analysis.json")
        with open(style_path, 'w') as f:
            json.dump(style_results, f, indent=2)
        
        print(f"📄 Style analysis saved: {style_path}")
        
        return style_results
    
    def create_comprehensive_report(self, faces, novelty_results, style_results, targets_met):
        """Create comprehensive report"""
        print(f"\n📋 CREATING COMPREHENSIVE REPORT")
        print("=" * 40)
        
        report = {
            'report_date': datetime.now().isoformat(),
            'faces_analyzed': len(faces),
            'faces_directory': self.faces_dir,
            'analysis_directory': self.output_dir,
            'novelty_analysis': novelty_results,
            'style_analysis': style_results if style_results else None,
            'targets_check': {
                'week1_targets_met': targets_met,
                'novelty_gt_50': novelty_results['novelty_percentage'] > 50,
                'similarity_lt_08': novelty_results['avg_similarity'] < 0.8
            },
            'files_generated': [
                'loading_summary.json',
                'novelty_results.json',
                'novelty_analysis.png',
                'target_check.json',
                'style_analysis.json (if available)',
                'comprehensive_report.json'
            ]
        }
        
        report_path = os.path.join(self.output_dir, "comprehensive_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create text summary
        summary = f"""
        EXISTING FACE ANALYSIS REPORT
        ==============================
        
        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Faces Directory: {self.faces_dir}
        Analysis Output: {self.output_dir}
        
        SUMMARY:
        - Faces analyzed: {len(faces)}
        - Unique faces: {novelty_results['n_unique']}
        - Novelty percentage: {novelty_results['novelty_percentage']:.1f}%
        - Average similarity: {novelty_results['avg_similarity']:.3f}
        - Maximum similarity: {novelty_results['max_similarity']:.3f}
        
        WEEK 1 TARGET CHECK:
        - Novelty > 50%: {'✅ YES' if novelty_results['novelty_percentage'] > 50 else '❌ NO'} ({novelty_results['novelty_percentage']:.1f}%)
        - Similarity < 0.8: {'✅ YES' if novelty_results['avg_similarity'] < 0.8 else '❌ NO'} ({novelty_results['avg_similarity']:.3f})
        - Both targets met: {'✅ YES' if targets_met else '❌ NO'}
        
        INTERPRETATION:
        {self.get_interpretation(novelty_results['novelty_percentage'], novelty_results['avg_similarity'])}
        
        FILES GENERATED:
        1. loading_summary.json - What was loaded
        2. novelty_results.json - Detailed novelty metrics
        3. novelty_analysis.png - Visualization
        4. target_check.json - Week 1 target assessment
        5. style_analysis.json - Style diversity metrics
        6. comprehensive_report.json - This report
        
        NEXT STEPS:
        1. Check the visualization: {self.output_dir}/novelty_analysis.png
        2. If targets not met, generate more faces with different seeds
        3. For full analysis: python ffhq_advanced_diversity_analysis_fixed_v2.py
        """
        
        summary_path = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"📄 Comprehensive report saved: {report_path}")
        print(f"📋 Text summary saved: {summary_path}")
        
        return report_path
    
    def get_interpretation(self, novelty_pct, avg_sim):
        """Get interpretation of results"""
        if novelty_pct > 90 and avg_sim < 0.5:
            return "EXCELLENT - Highly diverse and unique faces"
        elif novelty_pct > 70 and avg_sim < 0.7:
            return "GOOD - Good diversity, meets targets"
        elif novelty_pct > 50 and avg_sim < 0.8:
            return "MODERATE - Meets minimum targets"
        elif novelty_pct > 30:
            return "LOW - Some diversity but below targets"
        else:
            return "VERY LOW - Faces are too similar"
    
    def run_complete_analysis(self):
        """Run complete analysis on existing faces"""
        print("\n🔬 RUNNING COMPLETE ANALYSIS")
        print("=" * 50)
        
        # Step 1: Load existing faces
        faces = self.load_existing_faces()
        
        if len(faces) < 2:
            print("❌ Need at least 2 faces for analysis")
            return False
        
        print(f"\n📊 Starting analysis on {len(faces)} faces...")
        
        # Step 2: Analyze novelty
        novelty_results = self.analyze_novelty(faces)
        
        # Step 3: Analyze style diversity
        style_results = None
        if len(faces) >= 3:
            style_results = self.analyze_style_diversity(faces)
        
        # Step 4: Check Week 1 targets
        targets_met = self.check_week1_targets(novelty_results)
        
        # Step 5: Create comprehensive report
        report = self.create_comprehensive_report(faces, novelty_results, 
                                                 style_results, targets_met)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 All results saved in: {self.output_dir}")
        
        # Final message
        if novelty_results['novelty_percentage'] == 0:
            print(f"\n🚨 CRITICAL: 0% novelty - All faces are identical!")
            print(f"   This matches your original problem.")
            print(f"   SOLUTION: Use VERY different seeds (seed * 1000)")
        elif novelty_results['novelty_percentage'] < 50:
            print(f"\n⚠️  Warning: Low novelty ({novelty_results['novelty_percentage']:.1f}%)")
            print(f"   Try: Higher guidance scale (10-15) and more spaced seeds")
        else:
            print(f"\n🎉 Good news! You have {novelty_results['novelty_percentage']:.1f}% novelty")
            print(f"   (Was 0% in your original report)")
        
        return True

def main():
    """Main function"""
    print("📊 EXISTING FACE NOVELTY ANALYZER")
    print("=" * 60)
    print("\nThis script analyzes ALREADY GENERATED faces.")
    print("It calculates:")
    print("1. Novelty percentage (target: >50%)")
    print("2. Average similarity (target: <0.8)")
    print("3. Style diversity")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir', type=str,
                       help='Path to directory containing face images')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FaceAnalyzer(args.faces_dir)
    
    # Run analysis
    success = analyzer.run_complete_analysis()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    if success:
        print("📊 Check the results in the output directory!")
    else:
        print("❌ Analysis failed. Check error messages.")

if __name__ == "__main__":
    # Set matplotlib backend for display
    plt.switch_backend('TkAgg')  # or 'Qt5Agg' or 'Agg'
    
    # Run
    main()
