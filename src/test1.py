"""
Test script for feature matching and depth-weighted RANSAC.
Processes a single pair of images provided as command-line arguments.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Assuming your modules are importable
# Make sure the 'core' directory is in Python's path
# You might need to add: sys.path.append(str(Path(__file__).parent.parent))
from core.feature_matcher import FeatureMatcher
from core.depth_estimator import DepthEstimator
from core.homography import DepthAwareHomography


def test_feature_matching(img1, img2, matcher):
    """Test feature detection and matching"""
    print("\n=== 1. FEATURE MATCHING TEST ===")
    kp1, des1 = matcher.detect_and_compute(img1)
    kp2, des2 = matcher.detect_and_compute(img2)
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    matches = matcher.match_features(des1, des2)
    print(f"Found {len(matches)} matches after ratio test")

    stats = matcher.get_match_statistics(kp1, kp2, matches)
    print(f"Match ratio: {stats['match_ratio']:.3f}")
    print(f"Mean distance: {stats['mean_distance']:.2f}")

    pts1, pts2 = matcher.extract_matched_points(kp1, kp2, matches)
    vis_matches = matcher.visualize_matches(img1, kp1, img2, kp2, matches, max_display=100)
    return kp1, kp2, matches, pts1, pts2, vis_matches


def test_depth_estimation(pts1, pts2, depth_estimator):
    """Test sparse depth estimation from Sampson distance"""
    print("\n=== 2. DEPTH ESTIMATION TEST ===")
    F, mask = depth_estimator.compute_fundamental_matrix(pts1, pts2)
    n_inliers = mask.sum()
    print(f"Fundamental matrix inliers: {n_inliers}/{len(pts1)} ({n_inliers/len(pts1)*100:.1f}%)")

    sampson_distances = depth_estimator.compute_sampson_distance(pts1, pts2, F)
    depth_weights, F = depth_estimator.estimate_sparse_depth(pts1, pts2, F)
    print(f"Depth weights range: [{depth_weights.min():.3f}, {depth_weights.max():.3f}]")

    background_pts = (depth_weights > 0.7).sum()
    foreground_pts = (depth_weights < 0.4).sum()
    print(f"Background points (w > 0.7): {background_pts} ({background_pts/len(depth_weights)*100:.1f}%)")
    print(f"Foreground points (w < 0.4): {foreground_pts} ({foreground_pts/len(depth_weights)*100:.1f}%)")
    return F, sampson_distances, depth_weights


def test_homography_estimation(pts1, pts2, depth_weights, homography_estimator):
    """Test standard vs depth-weighted homography"""
    print("\n=== 3. HOMOGRAPHY ESTIMATION TEST ===")

    # Standard homography
    print("\n--- Standard RANSAC ---")
    H_standard, inliers_standard = homography_estimator.estimate(pts1, pts2, depth_weights=None)
    stats_standard = homography_estimator.get_inlier_statistics(pts1, pts2, H_standard, inliers_standard)
    print(f"Inliers: {stats_standard['n_inliers']}/{len(pts1)} ({stats_standard['inlier_ratio']*100:.1f}%)")
    print(f"Mean error: {stats_standard['mean_inlier_error']:.3f} pixels")

    # Depth-weighted homography
    print("\n--- Depth-Weighted RANSAC ---")
    H_weighted, inliers_weighted = homography_estimator.estimate(pts1, pts2, depth_weights=depth_weights)
    stats_weighted = homography_estimator.get_inlier_statistics(pts1, pts2, H_weighted, inliers_weighted)
    print(f"Inliers: {stats_weighted['n_inliers']}/{len(pts1)} ({stats_weighted['inlier_ratio']*100:.1f}%)")
    print(f"Mean error: {stats_weighted['mean_inlier_error']:.3f} pixels")

    # Compare inlier distributions
    print("\n--- Comparison ---")
    bg_inliers_std = ((depth_weights > 0.7) & inliers_standard).sum()
    bg_inliers_weighted = ((depth_weights > 0.7) & inliers_weighted).sum()
    print(f"Background inliers: Standard={bg_inliers_std}, Weighted={bg_inliers_weighted}")

    fg_inliers_std = ((depth_weights < 0.4) & inliers_standard).sum()
    fg_inliers_weighted = ((depth_weights < 0.4) & inliers_weighted).sum()
    print(f"Foreground inliers: Standard={fg_inliers_std}, Weighted={fg_inliers_weighted}")
    return H_standard, H_weighted, inliers_standard, inliers_weighted


def visualize_results(img1, img2, pts1, pts2, depth_weights,
                      inliers_standard, inliers_weighted,
                      H_standard, H_weighted, vis_matches, num_matches):
    """Create comprehensive visualization"""
    # Visualization 1: Feature Matches
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Feature Matches: {num_matches} found', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualization 2: Analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analysis of Standard vs. Depth-Weighted RANSAC', fontsize=20)
    
    # Depth weights visualization
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    scatter = ax.scatter(pts1[:, 0], pts1[:, 1], c=depth_weights,
                        cmap='viridis_r', s=25, alpha=0.8, vmin=0, vmax=1)
    ax.set_title('Estimated Depth Weights\n(Yellow=Near, Purple=Far)')
    plt.colorbar(scatter, ax=ax, label='Depth Weight (Near to Far)')
    ax.axis('off')

    # Depth weight distribution
    ax = axes[0, 1]
    ax.hist(depth_weights, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
    ax.axvline(0.7, color='purple', linestyle='--', label='Background Threshold')
    ax.axvline(0.4, color='orange', linestyle='--', label='Foreground Threshold')
    ax.set_xlabel('Depth Weight')
    ax.set_ylabel('Number of Points')
    ax.set_title('Depth Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Standard RANSAC inliers
    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax.scatter(pts1[~inliers_standard, 0], pts1[~inliers_standard, 1],
              c='red', s=15, alpha=0.6, label='Outliers')
    ax.scatter(pts1[inliers_standard, 0], pts1[inliers_standard, 1],
              c='lime', s=15, alpha=0.8, label='Inliers')
    ax.set_title(f'Standard RANSAC ({inliers_standard.sum()} inliers)')
    ax.legend()
    ax.axis('off')

    # Depth-weighted RANSAC inliers
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax.scatter(pts1[~inliers_weighted, 0], pts1[~inliers_weighted, 1],
              c='red', s=15, alpha=0.6, label='Outliers')
    ax.scatter(pts1[inliers_weighted, 0], pts1[inliers_weighted, 1],
              c='lime', s=15, alpha=0.8, label='Inliers')
    ax.set_title(f'Depth-Weighted RANSAC ({inliers_weighted.sum()} inliers)')
    ax.legend()
    ax.axis('off')

    # Warped image comparison - Standard
    h, w = img2.shape[:2]
    img2_warped_std = cv2.warpPerspective(img2, H_standard, (w, h))
    blend_std = cv2.addWeighted(img1, 0.5, img2_warped_std, 0.5, 0)
    ax = axes[1, 1]
    ax.imshow(cv2.cvtColor(blend_std, cv2.COLOR_BGR2RGB))
    ax.set_title('Standard Homography Blend')
    ax.axis('off')

    # Warped image comparison - Weighted
    img2_warped_weighted = cv2.warpPerspective(img2, H_weighted, (w, h))
    blend_weighted = cv2.addWeighted(img1, 0.5, img2_warped_weighted, 0.5, 0)
    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(blend_weighted, cv2.COLOR_BGR2RGB))
    ax.set_title('Depth-Weighted Homography Blend')
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # 1. Check for command-line arguments
    if len(sys.argv) != 3:
        print("❌ Usage: python test_script.py <path_to_image_1> <path_to_image_2>")
        sys.exit(1)

    img_path1 = Path(sys.argv[1])
    img_path2 = Path(sys.argv[2])

    # 2. Load images
    print(f"Loading images:\n- {img_path1.name}\n- {img_path2.name}")
    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))

    if img1 is None or img2 is None:
        print(f"❌ Error: Could not load one or both images. Check paths.")
        sys.exit(1)
        
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    # 3. Initialize components with desired parameters
    matcher = FeatureMatcher(nfeatures=5000, ratio_threshold=0.75)
    depth_estimator = DepthEstimator(alpha=3.0, percentile=95, rbf_sigma=20.0)
    homography_estimator = DepthAwareHomography(ransac_iterations=2000, inlier_threshold=3.0)

    # 4. Run the full pipeline
    # Test 1: Feature matching
    kp1, kp2, matches, pts1, pts2, vis_matches = test_feature_matching(img1, img2, matcher)
    if len(pts1) < 8:
        print("❌ ERROR: Not enough matches found to proceed. Try adjusting matcher parameters.")
        return

    # Test 2: Depth estimation
    F, sampson_distances, depth_weights = test_depth_estimation(pts1, pts2, depth_estimator)

    # Test 3: Homography estimation
    H_std, H_weighted, inliers_std, inliers_weighted = \
        test_homography_estimation(pts1, pts2, depth_weights, homography_estimator)

    # 5. Visualization
    print("\n=== 4. GENERATING VISUALIZATIONS ===")
    visualize_results(
        img1, img2, pts1, pts2, depth_weights,
        inliers_std, inliers_weighted,
        H_std, H_weighted,
        vis_matches, len(matches)
    )

    print("\n✅ === TEST COMPLETE ===")


if __name__ == "__main__":
    main()