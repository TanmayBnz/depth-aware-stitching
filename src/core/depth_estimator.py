"""
Sparse depth estimation from Sampson distance in epipolar geometry
IMPROVED VERSION: Uses homography residuals as primary depth indicator
"""

import cv2
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from typing import Tuple, Optional


class DepthEstimator:
    """
    Estimates sparse depth using multiple geometric cues:
    1. Homography residuals (primary)
    2. Sampson distance (secondary)
    3. Spatial coherence filtering
    """
    
    def __init__(self, alpha: float = 3.0, percentile: int = 95, rbf_sigma: float = 20.0):
        """
        Initialize depth estimator
        
        Args:
            alpha: Sensitivity parameter for depth weight computation
            percentile: Percentile for robust normalization
            rbf_sigma: Sigma for RBF interpolation
        """
        self.alpha = alpha
        self.percentile = percentile
        self.rbf_sigma = rbf_sigma
    
    def compute_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fundamental matrix using RANSAC
        
        Args:
            pts1, pts2
        
        Returns:
            F: Fundamental matrix (3, 3)
            mask: Inlier mask (n_points,)
        """
        if len(pts1) < 8:
            raise ValueError("Need at least 8 point correspondences")
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
        
        if F is None:
            # Fallback to 8-point algorithm
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        
        return F, mask.ravel().astype(bool) if mask is not None else np.ones(len(pts1), dtype=bool)
    
    def compute_sampson_distance(self, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Compute Sampson distance for each correspondence
        
        Sampson distance is a first-order geometric approximation:
        d = (x2^T F x1)^2 / (||F x1||^2 + ||F^T x2||^2)
        
        Args:
            pts1, pts2: Point correspondences (n_points, 2)
            F: Fundamental matrix (3, 3)
        
        Returns:
            distances: Sampson distances (n_points,)
        """
        # homogenous
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
        
        epipolar_errors = np.sum(pts2_h * (pts1_h @ F.T), axis=1)
        
        # Compute denominators
        Fx1 = (F @ pts1_h.T).T
        FTx2 = (F.T @ pts2_h.T).T
        norm_Fx1_sq = Fx1[:, 0]**2 + Fx1[:, 1]**2
        norm_FTx2_sq = FTx2[:, 0]**2 + FTx2[:, 1]**2
        
        # Sampson distance
        denominator = norm_Fx1_sq + norm_FTx2_sq
        denominator = np.maximum(denominator, 1e-10)
        
        sampson_distances = (epipolar_errors ** 2) / denominator
        
        return np.abs(sampson_distances)
    
    def estimate_sparse_depth(self, pts1: np.ndarray, pts2: np.ndarray,
                              F: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMPROVED: Estimate sparse depth weights using homography residuals + Sampson distance
        
        Key improvement: Use homography residual as primary depth indicator
        - Background points fit a single homography well (low residual)
        - Foreground points violate homography (high residual)
        
        Args:
            pts1, pts2: Matched point coordinates
            F: Fundamental matrix (if None, will be computed)
        
        Returns:
            depth_weights: Background weights (n_points,), range [0, 1]
                          1.0 = background (far), 0.0 = foreground (close)
            F: Fundamental matrix used
        """
        if F is None:
            F, _ = self.compute_fundamental_matrix(pts1, pts2)
        
        print("  [DepthEstimator] Computing depth using homography residuals...")
        
        # Method 1: Homography residuals (PRIMARY - most reliable)
        H_weights = self._estimate_from_homography_residuals(pts1, pts2)
        
        # Method 2: Sampson distance (SECONDARY - for validation)
        sampson_distances = self.compute_sampson_distance(pts1, pts2, F)
        S_weights = self._sampson_to_weights(sampson_distances)
        
        # Method 3: Spatial coherence (TERTIARY - outlier detection)
        coherence_scores = self._compute_spatial_coherence(pts1, H_weights)
        
        # Combine multiple cues with weighted average
        # Homography residuals are most reliable, so weight them highest
        depth_weights = (
            0.6 * H_weights +      # Primary: homography fit
            0.3 * S_weights +      # Secondary: epipolar constraint
            0.1 * coherence_scores # Tertiary: spatial smoothness
        )
        
        # Post-process: remove outliers and normalize
        depth_weights = self._post_process_weights(depth_weights, pts1)
        
        print(f"  [DepthEstimator] Depth weights: min={depth_weights.min():.3f}, "
              f"max={depth_weights.max():.3f}, mean={depth_weights.mean():.3f}")
        
        return depth_weights, F
    
    def _estimate_from_homography_residuals(self, pts1: np.ndarray, 
                                           pts2: np.ndarray) -> np.ndarray:
        """
        Estimate depth from homography residuals
        
        Key insight: Compute homography on ALL points, then use residuals
        - Low residual = fits homography = background (weight ~ 1.0)
        - High residual = violates homography = foreground (weight ~ 0.0)
        """
        # Fit homography to ALL points (not just inliers)
        # Use RANSAC for robustness, but we want residuals for all points
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 
                                     ransacReprojThreshold=5.0)
        
        if H is None:
            # Fallback: use least squares homography
            H, _ = cv2.findHomography(pts1, pts2, 0)
        
        # Compute residuals for ALL points
        residuals = self._compute_homography_residuals(pts1, pts2, H)
        
        # Convert residuals to weights using robust normalization
        # Use median absolute deviation (MAD) for robust scaling
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        
        if mad < 1e-6:
            # All points fit homography - probably pure rotation
            print("    [Warning] All points fit homography - may be pure rotation scene")
            return np.ones_like(residuals)
        
        # Normalize using MAD (more robust than percentile)
        normalized_residuals = (residuals - median_residual) / (mad + 1e-6)
        
        # Clip to prevent overflow in exp()
        normalized_residuals = np.clip(normalized_residuals, -10, 10)
        
        # Convert to weights: low residual = background = high weight
        # Use sigmoid for smooth transition
        depth_weights = 1.0 / (1.0 + np.exp(normalized_residuals - 2.0))
        
        return depth_weights
    
    def _compute_homography_residuals(self, pts1: np.ndarray, 
                                      pts2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Compute symmetric reprojection error"""
        # Forward: pts1 -> pts2
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts1_projected = (H @ pts1_h.T).T
        pts1_projected = pts1_projected[:, :2] / (pts1_projected[:, 2:3] + 1e-8)
        forward_error = np.linalg.norm(pts2 - pts1_projected, axis=1)
        
        # Backward: pts2 -> pts1
        try:
            H_inv = np.linalg.inv(H)
            pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
            pts2_projected = (H_inv @ pts2_h.T).T
            pts2_projected = pts2_projected[:, :2] / (pts2_projected[:, 2:3] + 1e-8)
            backward_error = np.linalg.norm(pts1 - pts2_projected, axis=1)
            
            # Use symmetric error (more robust)
            residuals = (forward_error + backward_error) / 2.0
        except:
            residuals = forward_error
        
        return residuals
    
    def _sampson_to_weights(self, sampson_distances: np.ndarray) -> np.ndarray:
        """Convert Sampson distances to weights with robust normalization"""
        # Robust normalization using percentile
        d_max = np.percentile(sampson_distances, self.percentile)
        if d_max < 1e-6:
            d_max = sampson_distances.max() + 1e-6
        
        normalized = np.clip(sampson_distances / d_max, 0, 3)
        weights = 1.0 / (1.0 + self.alpha * normalized)
        
        return weights
    
    def _compute_spatial_coherence(self, pts: np.ndarray, 
                                   weights: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Compute spatial coherence scores to detect outliers
        
        Idea: Nearby points should have similar depth
        Points with very different weights from neighbors are likely outliers
        """
        if len(pts) < k:
            return np.ones_like(weights)
        
        # Build KD-tree for nearest neighbor search
        tree = cKDTree(pts)
        
        coherence_scores = np.zeros_like(weights)
        
        for i in range(len(pts)):
            # Find k nearest neighbors
            distances, indices = tree.query(pts[i], k=min(k+1, len(pts)))
            neighbor_indices = indices[1:] if len(indices) > 1 else indices  # Exclude self
            
            if len(neighbor_indices) == 0:
                coherence_scores[i] = 1.0
                continue
            
            # Compute weight consistency with neighbors
            neighbor_weights = weights[neighbor_indices]
            weight_diff = np.abs(weights[i] - neighbor_weights)
            
            # High coherence = similar to neighbors
            coherence_scores[i] = 1.0 - np.mean(weight_diff)
        
        return np.clip(coherence_scores, 0, 1)
    
    def _post_process_weights(self, weights: np.ndarray, 
                             pts: np.ndarray) -> np.ndarray:
        """
        Post-process weights to remove outliers and smooth
        """
        # Clip to valid range
        weights = np.clip(weights, 0, 1)
        
        # Detect and handle extreme outliers (weights very far from median)
        median_weight = np.median(weights)
        mad = np.median(np.abs(weights - median_weight))
        
        if mad > 0.01:  # Only filter if there's significant variation
            outlier_mask = np.abs(weights - median_weight) > 5 * mad
            if outlier_mask.sum() > 0:
                # Replace outliers with local median
                tree = cKDTree(pts)
                
                for i in np.where(outlier_mask)[0]:
                    distances, indices = tree.query(pts[i], k=min(10, len(pts)))
                    neighbor_weights = weights[indices[~outlier_mask[indices]]]
                    if len(neighbor_weights) > 0:
                        weights[i] = np.median(neighbor_weights)
        
        return weights
    
    def interpolate_to_dense(self, pts: np.ndarray, depth_weights: np.ndarray, image_shape: Tuple[int, int], method: str = 'rbf') -> np.ndarray:
        """
        Interpolate sparse depth to dense depth map
        
        Args:
            pts: Point coordinates (n_points, 2)
            depth_weights: Depth values at points (n_points,)
            image_shape: (height, width) of output
            method: 'rbf' or 'nearest'
        
        Returns:
            depth_map: Dense depth map (height, width)
        """
        h, w = image_shape[:2]

        valid_mask = (depth_weights > 0) & np.isfinite(depth_weights)
        valid_mask &= (pts[:, 0] >= 0) & (pts[:, 0] < w)
        valid_mask &= (pts[:, 1] >= 0) & (pts[:, 1] < h)
        
        pts_valid = pts[valid_mask]
        weights_valid = depth_weights[valid_mask]
        
        if len(pts_valid) < 10:
            print("Warning: Not enough valid points for interpolation, returning uniform depth")
            return np.ones((h, w), dtype=np.float32) * 0.5
        
        if method == 'rbf':
            try:
                depth_map = self._interpolate_rbf(pts_valid, weights_valid, (h, w))
            except Exception as e:
                print(f"RBF interpolation failed: {e}, falling back to nearest neighbor")
                depth_map = self._interpolate_nearest(pts_valid, weights_valid, (h, w))
        else:
            depth_map = self._interpolate_nearest(pts_valid, weights_valid, (h, w))
        
        return depth_map
    
    def _interpolate_rbf(self, pts: np.ndarray, values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        RBF interpolation
        """
        h, w = shape
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        
        rbf = Rbf(x_coords, y_coords, values, function='gaussian', smooth=self.rbf_sigma)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        depth_map = rbf(grid_x, grid_y)
        depth_map = np.clip(depth_map, 0, 1)
        
        return depth_map.astype(np.float32)
    
    def _interpolate_nearest(self, pts: np.ndarray, values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Nearest neighbor interpolation (fast but less smooth)
        """
        h, w = shape
        
        # Build KD-tree
        tree = cKDTree(pts)
        
        # Query for each pixel
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        distances, indices = tree.query(grid_pts, k=1)
        
        # Get values
        depth_map = values[indices].reshape(h, w)
        
        return depth_map.astype(np.float32)
    
    def visualize_depth(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Create visualization of depth map
        
        Args:
            depth_map: Depth map (height, width), range [0, 1]
            colormap: OpenCV colormap
        
        Returns:
            vis: Colored depth visualization (height, width, 3)
        """
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        vis = cv2.applyColorMap(depth_uint8, colormap)
        
        return vis