"""
Enhanced test script with comprehensive comparison metrics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from core.feature_matcher import FeatureMatcher
from core.depth_estimator import DepthEstimator
from core.homography import DepthAwareHomography


def compute_alignment_quality(img1, img2, H, overlap_mask=None):
    """
    Compute image alignment quality metrics
    """
    h, w = img2.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (w, h))
    
    if overlap_mask is None:
        # Use center region as overlap
        overlap_mask = np.zeros((h, w), dtype=bool)
        overlap_mask[h//4:3*h//4, w//4:3*w//4] = True
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_warped = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    
    # Compute differences in overlap region
    diff = np.abs(gray1.astype(float) - gray2_warped.astype(float))
    
    metrics = {
        'mean_absolute_error': diff[overlap_mask].mean(),
        'median_absolute_error': np.median(diff[overlap_mask]),
        'std_error': diff[overlap_mask].std(),
        'psnr': cv2.PSNR(gray1, gray2_warped)
    }
    
    return metrics, diff


def compute_background_alignment_quality(img1, img2, H, pts1, depth_weights, threshold=0.7):
    """
    Measure alignment quality specifically for background regions
    """
    h, w = img2.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (w, h))
    
    # Create mask for background regions
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    bg_points = pts1[depth_weights > threshold].astype(int)
    
    # Dilate around background points
    for pt in bg_points:
        x, y = pt
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(bg_mask, (x, y), 30, 255, -1)
    
    bg_mask = bg_mask > 0
    
    if bg_mask.sum() == 0:
        return {'error': float('inf')}
    
    # Compute error in background regions
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_warped = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray1.astype(float) - gray2_warped.astype(float))
    
    return {
        'bg_mean_error': diff[bg_mask].mean(),
        'bg_median_error': np.median(diff[bg_mask]),
        'bg_coverage': bg_mask.sum() / (h * w)
    }


def test_homography_comparison(pts1, pts2, depth_weights, homography_estimator, img1, img2):
    """
    Comprehensive comparison of standard vs depth-weighted homography
    """
    print("\n=== 3. HOMOGRAPHY ESTIMATION TEST ===")

    # Standard homography
    print("\n--- Standard RANSAC ---")
    H_standard, inliers_standard = homography_estimator.estimate(pts1, pts2, depth_weights=None)
    stats_standard = homography_estimator.get_inlier_statistics(
        pts1, pts2, H_standard, inliers_standard, depth_weights
    )
    
    print(f"Inliers: {stats_standard['n_inliers']}/{len(pts1)} ({stats_standard['inlier_ratio']*100:.1f}%)")
    print(f"Mean error: {stats_standard['mean_inlier_error']:.3f} pixels")
    print(f"Background inliers: {stats_standard['bg_inliers']} ({stats_standard['bg_inlier_ratio']*100:.1f}%)")
    print(f"Mean inlier depth weight: {stats_standard['mean_inlier_depth_weight']:.3f}")
    
    # Depth-weighted homography
    print("\n--- Depth-Weighted RANSAC ---")
    H_weighted, inliers_weighted = homography_estimator.estimate(pts1, pts2, depth_weights=depth_weights)
    stats_weighted = homography_estimator.get_inlier_statistics(
        pts1, pts2, H_weighted, inliers_weighted, depth_weights
    )
    
    print(f"Inliers: {stats_weighted['n_inliers']}/{len(pts1)} ({stats_weighted['inlier_ratio']*100:.1f}%)")
    print(f"Mean error: {stats_weighted['mean_inlier_error']:.3f} pixels")
    print(f"Background inliers: {stats_weighted['bg_inliers']} ({stats_weighted['bg_inlier_ratio']*100:.1f}%)")
    print(f"Mean inlier depth weight: {stats_weighted['mean_inlier_depth_weight']:.3f}")
    
    # Debug info
    debug_info = homography_estimator.get_debug_info()
    if debug_info:
        print(f"\nSampling statistics:")
        print(f"  Background sample ratio: {debug_info['bg_sample_ratio']:.3f} (expected: {debug_info['expected_bg_ratio']:.3f})")
    
    # Image-based quality metrics
    print("\n--- Image Alignment Quality ---")
    qual_std, diff_std = compute_alignment_quality(img1, img2, H_standard)
    qual_weighted, diff_weighted = compute_alignment_quality(img1, img2, H_weighted)
    
    print(f"Standard RANSAC:")
    print(f"  MAE: {qual_std['mean_absolute_error']:.2f}, PSNR: {qual_std['psnr']:.2f} dB")
    print(f"Depth-Weighted RANSAC:")
    print(f"  MAE: {qual_weighted['mean_absolute_error']:.2f}, PSNR: {qual_weighted['psnr']:.2f} dB")
    
    # Background-specific quality
    bg_qual_std = compute_background_alignment_quality(img1, img2, H_standard, pts1, depth_weights)
    bg_qual_weighted = compute_background_alignment_quality(img1, img2, H_weighted, pts1, depth_weights)
    
    print(f"\n--- Background Region Alignment (KEY METRIC) ---")
    print(f"Standard RANSAC background error: {bg_qual_std['bg_mean_error']:.2f}")
    print(f"Depth-Weighted background error: {bg_qual_weighted['bg_mean_error']:.2f}")
    
    improvement = (bg_qual_std['bg_mean_error'] - bg_qual_weighted['bg_mean_error']) / bg_qual_std['bg_mean_error'] * 100
    if improvement > 0:
        print(f"✅ Depth-weighted is {improvement:.1f}% better for background!")
    else:
        print(f"❌ Depth-weighted is {-improvement:.1f}% worse for background")
    
    return (H_standard, H_weighted, inliers_standard, inliers_weighted, 
            diff_std, diff_weighted, stats_standard, stats_weighted)


def visualize_comparison(img1, img2, pts1, pts2, depth_weights,
                        inliers_standard, inliers_weighted,
                        H_standard, H_weighted, 
                        diff_std, diff_weighted,
                        stats_std, stats_weighted,
                        vis_matches, num_matches):
    """Enhanced visualization with side-by-side comparison"""
    
    # Figure 1: Feature analysis
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Feature Matches: {num_matches} found', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('output_matches.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 2: Comprehensive comparison
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Standard vs Depth-Weighted RANSAC Comparison', fontsize=20, fontweight='bold')
    
    # Row 1: Depth visualization
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    scatter = ax.scatter(pts1[:, 0], pts1[:, 1], c=depth_weights,
                        cmap='viridis_r', s=30, alpha=0.8, vmin=0, vmax=1, edgecolors='white', linewidths=0.5)
    ax.set_title('Estimated Depth\n(Yellow=Near, Purple=Far)', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Background Weight')
    ax.axis('off')

    # Inlier comparison
    ax = axes[0, 1]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax.scatter(pts1[~inliers_standard, 0], pts1[~inliers_standard, 1],
              c='red', s=20, alpha=0.5, label='Outliers')
    ax.scatter(pts1[inliers_standard, 0], pts1[inliers_standard, 1],
              c='lime', s=20, alpha=0.8, label='Inliers')
    ax.set_title(f'Standard RANSAC\n{stats_std["n_inliers"]} inliers (bg weight: {stats_std["mean_inlier_depth_weight"]:.2f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax.scatter(pts1[~inliers_weighted, 0], pts1[~inliers_weighted, 1],
              c='red', s=20, alpha=0.5, label='Outliers')
    ax.scatter(pts1[inliers_weighted, 0], pts1[inliers_weighted, 1],
              c='lime', s=20, alpha=0.8, label='Inliers')
    ax.set_title(f'Depth-Weighted RANSAC\n{stats_weighted["n_inliers"]} inliers (bg weight: {stats_weighted["mean_inlier_depth_weight"]:.2f})', fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')

    # Row 2: Warped overlays
    h, w = img2.shape[:2]
    img2_warped_std = cv2.warpPerspective(img2, H_standard, (w, h))
    img2_warped_weighted = cv2.warpPerspective(img2, H_weighted, (w, h))
    
    blend_std = cv2.addWeighted(img1, 0.5, img2_warped_std, 0.5, 0)
    blend_weighted = cv2.addWeighted(img1, 0.5, img2_warped_weighted, 0.5, 0)
    
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax.set_title('Original Image 1', fontsize=12)
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(cv2.cvtColor(blend_std, cv2.COLOR_BGR2RGB))
    ax.set_title('Standard Homography Overlay', fontsize=12)
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(blend_weighted, cv2.COLOR_BGR2RGB))
    ax.set_title('Depth-Weighted Overlay', fontsize=12)
    ax.axis('off')

    # Row 3: Error visualization
    ax = axes[2, 0]
    im = ax.imshow(diff_std, cmap='hot', vmin=0, vmax=50)
    ax.set_title('Standard RANSAC Error Map', fontsize=12)
    plt.colorbar(im, ax=ax, label='Pixel Difference')
    ax.axis('off')

    ax = axes[2, 1]
    im = ax.imshow(diff_weighted, cmap='hot', vmin=0, vmax=50)
    ax.set_title('Depth-Weighted Error Map', fontsize=12)
    plt.colorbar(im, ax=ax, label='Pixel Difference')
    ax.axis('off')
    
    # Difference of differences
    ax = axes[2, 2]
    diff_comparison = diff_std - diff_weighted
    im = ax.imshow(diff_comparison, cmap='RdYlGn', vmin=-20, vmax=20)
    ax.set_title('Improvement Map\n(Green=Better, Red=Worse)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Error Reduction')
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('output_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("❌ Usage: python test_enhanced.py <path_to_image_1> <path_to_image_2>")
        sys.exit(1)

    img_path1 = Path(sys.argv[1])
    img_path2 = Path(sys.argv[2])

    print(f"Loading images:\n- {img_path1.name}\n- {img_path2.name}")
    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))

    if img1 is None or img2 is None:
        print(f"❌ Error: Could not load one or both images.")
        sys.exit(1)
        
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    # Initialize
    matcher = FeatureMatcher(nfeatures=5000, ratio_threshold=0.75)
    depth_estimator = DepthEstimator(alpha=3.0, percentile=95, rbf_sigma=20.0)
    homography_estimator = DepthAwareHomography(ransac_iterations=2000, inlier_threshold=3.0)

    # Feature matching
    print("\n=== 1. FEATURE MATCHING TEST ===")
    kp1, des1 = matcher.detect_and_compute(img1)
    kp2, des2 = matcher.detect_and_compute(img2)
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    matches = matcher.match_features(des1, des2)
    print(f"Found {len(matches)} matches")

    pts1, pts2 = matcher.extract_matched_points(kp1, kp2, matches)
    vis_matches = matcher.visualize_matches(img1, kp1, img2, kp2, matches, max_display=100)
    
    if len(pts1) < 8:
        print("❌ ERROR: Not enough matches")
        return

    # Depth estimation
    print("\n=== 2. DEPTH ESTIMATION TEST ===")
    depth_weights, F = depth_estimator.estimate_sparse_depth(pts1, pts2)
    print(f"Background points (w > 0.7): {(depth_weights > 0.7).sum()} ({(depth_weights > 0.7).sum()/len(depth_weights)*100:.1f}%)")
    print(f"Foreground points (w < 0.4): {(depth_weights < 0.4).sum()} ({(depth_weights < 0.4).sum()/len(depth_weights)*100:.1f}%)")

    # Homography estimation with comprehensive comparison
    results = test_homography_comparison(pts1, pts2, depth_weights, homography_estimator, img1, img2)
    H_std, H_weighted, inliers_std, inliers_weighted, diff_std, diff_weighted, stats_std, stats_weighted = results

    # Visualization
    print("\n=== 4. GENERATING VISUALIZATIONS ===")
    visualize_comparison(
        img1, img2, pts1, pts2, depth_weights,
        inliers_std, inliers_weighted,
        H_std, H_weighted,
        diff_std, diff_weighted,
        stats_std, stats_weighted,
        vis_matches, len(matches)
    )

    print("\n✅ === TEST COMPLETE ===")
    print("Check output_matches.png and output_comparison.png for visualizations")


if __name__ == "__main__":
    main()