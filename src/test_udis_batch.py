"""
Batch testing on UDIS-D dataset with comprehensive evaluation
FIXED: JSON serialization and overflow issues
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from core.feature_matcher import FeatureMatcher
from core.depth_estimator import DepthEstimator
from core.homography import DepthAwareHomography


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


class UDISDEvaluator:
    """
    Evaluate depth-aware stitching on UDIS-D dataset
    """
    
    def __init__(self, output_dir="results/udis_d"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.matcher = FeatureMatcher(nfeatures=5000, ratio_threshold=0.75)
        self.depth_estimator = DepthEstimator(alpha=3.0, percentile=95, rbf_sigma=20.0)
        self.homography_estimator = DepthAwareHomography(ransac_iterations=2000, inlier_threshold=3.0)
        
        self.results = []
    
    def load_image_pair(self, img1_path, img2_path):
        """Load and validate image pair"""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images: {img1_path}, {img2_path}")
        
        return img1, img2
    
    def analyze_scene_parallax(self, depth_weights):
        """
        Categorize scene by parallax level
        ADJUSTED: More lenient thresholds for real-world datasets
        """
        depth_std = float(np.std(depth_weights))
        depth_range = float(depth_weights.max() - depth_weights.min())
        fg_ratio = float((depth_weights < 0.5).sum() / len(depth_weights))
        bg_ratio = float((depth_weights > 0.7).sum() / len(depth_weights))
        
        # Adjusted thresholds based on observation
        # UDIS-D typically has more subtle parallax
        if depth_std < 0.08:
            scene_type = "low_parallax"
        elif depth_std < 0.15:
            scene_type = "medium_parallax"
        else:
            scene_type = "high_parallax"
        
        return {
            'scene_type': scene_type,
            'depth_std': depth_std,
            'depth_range': depth_range,
            'fg_ratio': fg_ratio,
            'bg_ratio': bg_ratio
        }
    
    def compute_alignment_metrics(self, img1, img2, H):
        """Compute alignment quality metrics"""
        h, w = img2.shape[:2]
        img2_warped = cv2.warpPerspective(img2, H, (w, h))
        
        # Compute overlap region
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2_warped = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
        
        # Use center region as overlap
        overlap_mask = np.zeros((h, w), dtype=bool)
        overlap_mask[h//4:3*h//4, w//4:3*w//4] = True
        
        # Compute error
        diff = np.abs(gray1.astype(float) - gray2_warped.astype(float))
        
        metrics = {
            'mae': float(diff[overlap_mask].mean()),
            'median_ae': float(np.median(diff[overlap_mask])),
            'std_error': float(diff[overlap_mask].std()),
            'max_error': float(diff[overlap_mask].max())
        }
        
        # Try to compute PSNR (may fail if images are identical)
        try:
            metrics['psnr'] = float(cv2.PSNR(gray1, gray2_warped))
        except:
            metrics['psnr'] = 0.0
        
        return metrics
    
    def compute_background_metrics(self, img1, img2, H, pts1, depth_weights, threshold=0.7):
        """Compute metrics specifically for background regions"""
        h, w = img2.shape[:2]
        img2_warped = cv2.warpPerspective(img2, H, (w, h))
        
        # Create background mask
        bg_mask = np.zeros((h, w), dtype=np.uint8)
        bg_points = pts1[depth_weights > threshold].astype(int)
        
        for pt in bg_points:
            x, y = pt
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(bg_mask, (x, y), 30, 255, -1)
        
        bg_mask = bg_mask > 0
        
        if bg_mask.sum() == 0:
            return {'bg_mae': float('inf'), 'bg_coverage': 0.0}
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2_warped = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray1.astype(float) - gray2_warped.astype(float))
        
        return {
            'bg_mae': float(diff[bg_mask].mean()),
            'bg_median_ae': float(np.median(diff[bg_mask])),
            'bg_coverage': float(bg_mask.sum() / (h * w))
        }
    
    def evaluate_pair(self, img1_path, img2_path, pair_name):
        """Evaluate single image pair"""
        result = {
            'pair_name': pair_name,
            'img1_path': img1_path,
            'img2_path': img2_path,
            'success': False
        }
        
        try:
            # Load images
            img1, img2 = self.load_image_pair(img1_path, img2_path)
            result['image_size'] = list(img1.shape)
            
            # Feature matching
            kp1, des1 = self.matcher.detect_and_compute(img1)
            kp2, des2 = self.matcher.detect_and_compute(img2)
            matches = self.matcher.match_features(des1, des2)
            pts1, pts2 = self.matcher.extract_matched_points(kp1, kp2, matches)
            
            result['n_keypoints_1'] = int(len(kp1))
            result['n_keypoints_2'] = int(len(kp2))
            result['n_matches'] = int(len(matches))
            
            if len(pts1) < 8:
                result['error'] = 'Not enough matches'
                return result
            
            # Depth estimation
            depth_weights, F = self.depth_estimator.estimate_sparse_depth(pts1, pts2)
            scene_info = self.analyze_scene_parallax(depth_weights)
            result['scene_info'] = scene_info
            
            # Standard RANSAC
            H_std, inliers_std = self.homography_estimator.estimate(pts1, pts2, depth_weights=None)
            stats_std = self.homography_estimator.get_inlier_statistics(
                pts1, pts2, H_std, inliers_std, depth_weights
            )
            
            # Depth-weighted RANSAC
            H_weighted, inliers_weighted = self.homography_estimator.estimate(
                pts1, pts2, depth_weights=depth_weights
            )
            stats_weighted = self.homography_estimator.get_inlier_statistics(
                pts1, pts2, H_weighted, inliers_weighted, depth_weights
            )
            
            # Compute alignment metrics
            align_std = self.compute_alignment_metrics(img1, img2, H_std)
            align_weighted = self.compute_alignment_metrics(img1, img2, H_weighted)
            
            # Background-specific metrics
            bg_std = self.compute_background_metrics(img1, img2, H_std, pts1, depth_weights)
            bg_weighted = self.compute_background_metrics(img1, img2, H_weighted, pts1, depth_weights)
            
            # Convert stats to plain Python types
            stats_std_clean = convert_to_python_types(stats_std)
            stats_weighted_clean = convert_to_python_types(stats_weighted)
            
            # Store results
            result['standard'] = {
                'inlier_stats': stats_std_clean,
                'alignment': align_std,
                'background': bg_std
            }
            
            result['depth_weighted'] = {
                'inlier_stats': stats_weighted_clean,
                'alignment': align_weighted,
                'background': bg_weighted
            }
            
            # Compute improvements
            mae_red = float(align_std['mae'] - align_weighted['mae'])
            bg_mae_red = float(bg_std['bg_mae'] - bg_weighted['bg_mae']) if bg_std['bg_mae'] != float('inf') else 0.0
            
            result['improvements'] = {
                'mae_reduction': mae_red,
                'mae_reduction_pct': float((mae_red / align_std['mae'] * 100)) if align_std['mae'] > 0 else 0.0,
                'bg_mae_reduction': bg_mae_red,
                'bg_mae_reduction_pct': float((bg_mae_red / bg_std['bg_mae'] * 100)) if bg_std['bg_mae'] not in [0, float('inf')] else 0.0,
                'inlier_increase': int(stats_weighted_clean['n_inliers'] - stats_std_clean['n_inliers']),
                'bg_inlier_increase': int(stats_weighted_clean['bg_inliers'] - stats_std_clean['bg_inliers'])
            }
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error processing {pair_name}: {e}")
        
        return result
    
    def run_batch_evaluation(self, pairs_file=None, data_dir="data/raw/dataset/UDIS-D", max_pairs=None):
        """
        Run evaluation on all UDIS-D pairs
        
        Args:
            pairs_file: Optional file listing pairs
            data_dir: UDIS-D dataset directory
            max_pairs: Maximum number of pairs to evaluate (None = all)
        """
        # Get pairs
        if pairs_file and Path(pairs_file).exists():
            pairs = self._load_pairs_from_file(pairs_file)
        else:
            pairs = self._get_pairs_from_directory(data_dir)
        
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {len(pairs)} image pairs from UDIS-D")
        print(f"{'='*60}\n")
        
        # Process each pair
        for i, (img1_path, img2_path) in enumerate(tqdm(pairs, desc="Processing pairs")):
            pair_name = f"pair_{i+1:03d}"
            result = self.evaluate_pair(img1_path, img2_path, pair_name)
            self.results.append(result)
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def _get_pairs_from_directory(self, data_dir):
        """Get pairs from UDIS-D directory structure"""
        data_path = Path(data_dir)
        input1_path = data_path / "testing" / "input1"
        input2_path = data_path / "testing" / "input2"
        
        if not (input1_path.exists() and input2_path.exists()):
            raise ValueError(f"UDIS-D dataset not found at {data_dir}")
        
        images1 = sorted(input1_path.glob("*.jpg"))
        images2 = sorted(input2_path.glob("*.jpg"))
        
        # Handle mismatch in image counts
        min_len = min(len(images1), len(images2))
        
        return [(str(images1[i]), str(images2[i])) for i in range(min_len)]
    
    def _load_pairs_from_file(self, pairs_file):
        """Load pairs from text file"""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        return pairs
    
    def generate_summary(self):
        """Generate summary statistics and visualizations"""
        successful = [r for r in self.results if r.get('success', False)]
        
        if not successful:
            print("❌ No successful evaluations")
            return
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY ({len(successful)}/{len(self.results)} successful)")
        print(f"{'='*60}\n")
        
        # Categorize by scene type
        by_scene = {'low_parallax': [], 'medium_parallax': [], 'high_parallax': []}
        for r in successful:
            scene_type = r['scene_info']['scene_type']
            by_scene[scene_type].append(r)
        
        print("Scene Distribution:")
        for scene_type, results in by_scene.items():
            if results:
                depth_stds = [r['scene_info']['depth_std'] for r in results]
                print(f"  {scene_type}: {len(results)} pairs (avg depth_std: {np.mean(depth_stds):.3f})")
        
        # Overall statistics
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE")
        print("="*60)
        
        mae_reductions = [r['improvements']['mae_reduction'] for r in successful]
        bg_reductions = [r['improvements']['bg_mae_reduction'] for r in successful 
                        if r['improvements']['bg_mae_reduction'] != 0]
        
        print(f"\nMAE Reduction:")
        print(f"  Mean: {np.mean(mae_reductions):.2f} pixels ({np.mean(mae_reductions)/np.mean([r['standard']['alignment']['mae'] for r in successful])*100:.1f}%)")
        print(f"  Median: {np.median(mae_reductions):.2f} pixels")
        print(f"  Std: {np.std(mae_reductions):.2f} pixels")
        print(f"  Improved: {sum(1 for x in mae_reductions if x > 0)}/{len(mae_reductions)}")
        print(f"  Degraded: {sum(1 for x in mae_reductions if x < -1)}/{len(mae_reductions)}")
        
        if bg_reductions:
            print(f"\nBackground MAE Reduction:")
            print(f"  Mean: {np.mean(bg_reductions):.2f} pixels")
            print(f"  Median: {np.median(bg_reductions):.2f} pixels")
            print(f"  Std: {np.std(bg_reductions):.2f} pixels")
            print(f"  Improved: {sum(1 for x in bg_reductions if x > 0)}/{len(bg_reductions)}")
        
        # Per-scene-type analysis
        print("\n" + "="*60)
        print("PER-SCENE-TYPE ANALYSIS")
        print("="*60)
        
        for scene_type, results in by_scene.items():
            if not results:
                continue
            
            mae_red = [r['improvements']['mae_reduction'] for r in results]
            bg_red = [r['improvements']['bg_mae_reduction'] for r in results 
                     if r['improvements']['bg_mae_reduction'] != 0]
            
            print(f"\n{scene_type.upper().replace('_', ' ')} ({len(results)} pairs):")
            print(f"  MAE reduction: {np.mean(mae_red):.2f} ± {np.std(mae_red):.2f} pixels")
            if bg_red:
                print(f"  BG MAE reduction: {np.mean(bg_red):.2f} ± {np.std(bg_red):.2f} pixels")
            improved = sum(1 for x in mae_red if x > 0)
            degraded = sum(1 for x in mae_red if x < -1)
            print(f"  Improved: {improved}/{len(mae_red)}, Degraded: {degraded}/{len(mae_red)}")
        
        # Save detailed results
        try:
            results_file = self.output_dir / "detailed_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)  # Use default=str as fallback
            print(f"\n✅ Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save JSON results: {e}")
        
        # Generate plots
        self.plot_results(successful, by_scene)
    
    def plot_results(self, results, by_scene):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UDIS-D Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: MAE reduction histogram
        ax = axes[0, 0]
        mae_reductions = [r['improvements']['mae_reduction'] for r in results]
        ax.hist(mae_reductions, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.axvline(np.mean(mae_reductions), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(mae_reductions):.2f}')
        ax.set_xlabel('MAE Reduction (pixels)', fontsize=11)
        ax.set_ylabel('Number of Pairs', fontsize=11)
        ax.set_title('Overall MAE Reduction Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Background MAE reduction
        ax = axes[0, 1]
        bg_reductions = [r['improvements']['bg_mae_reduction'] for r in results 
                        if r['improvements']['bg_mae_reduction'] != 0]
        if bg_reductions:
            ax.hist(bg_reductions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
            ax.axvline(np.mean(bg_reductions), color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(bg_reductions):.2f}')
            ax.set_xlabel('Background MAE Reduction (pixels)', fontsize=11)
            ax.set_ylabel('Number of Pairs', fontsize=11)
            ax.set_title('Background Region MAE Reduction', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No background data', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3: Per-scene-type comparison
        ax = axes[1, 0]
        scene_types = []
        scene_means = []
        scene_stds = []
        scene_counts = []
        
        for scene_type, scene_results in by_scene.items():
            if scene_results:
                mae_red = [r['improvements']['mae_reduction'] for r in scene_results]
                scene_types.append(scene_type.replace('_', '\n').title())
                scene_means.append(np.mean(mae_red))
                scene_stds.append(np.std(mae_red))
                scene_counts.append(len(scene_results))
        
        if scene_types:
            x_pos = np.arange(len(scene_types))
            bars = ax.bar(x_pos, scene_means, yerr=scene_stds, alpha=0.7, capsize=5, edgecolor='black')
            
            # Color bars based on improvement
            colors = ['green' if m > 0 else 'red' for m in scene_means]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scene_types, fontsize=10)
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('Mean MAE Reduction (pixels)', fontsize=11)
            ax.set_title('Performance by Scene Type', fontsize=12)
            
            # Add count labels on bars
            for i, (mean, count) in enumerate(zip(scene_means, scene_counts)):
                ax.text(i, mean + (scene_stds[i] if mean > 0 else -scene_stds[i]), 
                       f'n={count}', ha='center', va='bottom' if mean > 0 else 'top', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Scatter - depth std vs improvement
        ax = axes[1, 1]
        depth_stds = [r['scene_info']['depth_std'] for r in results]
        mae_reds = [r['improvements']['mae_reduction'] for r in results]
        
        # Color by scene type
        colors_map = {'low_parallax': 'blue', 'medium_parallax': 'orange', 'high_parallax': 'red'}
        colors = [colors_map[r['scene_info']['scene_type']] for r in results]
        
        scatter = ax.scatter(depth_stds, mae_reds, c=colors, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No improvement')
        ax.set_xlabel('Depth Std (parallax indicator)', fontsize=11)
        ax.set_ylabel('MAE Reduction (pixels)', fontsize=11)
        ax.set_title('Improvement vs Scene Parallax', fontsize=12)
        
        # Add legend for scene types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=stype.replace('_', ' ').title()) 
                          for stype, color in colors_map.items()]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plots saved to: {plot_file}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate on UDIS-D dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/dataset/UDIS-D',
                       help='Path to UDIS-D dataset')
    parser.add_argument('--output_dir', type=str, default='results/udis_d',
                       help='Output directory for results')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum number of pairs to evaluate')
    parser.add_argument('--pairs_file', type=str, default=None,
                       help='Optional file listing specific pairs to test')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = UDISDEvaluator(output_dir=args.output_dir)
    
    # Run evaluation
    results = evaluator.run_batch_evaluation(
        pairs_file=args.pairs_file,
        data_dir=args.data_dir,
        max_pairs=args.max_pairs
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()