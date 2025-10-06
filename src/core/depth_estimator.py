"""
Sparse depth estimation from Sampson distance in epipolar geometry
"""

import cv2
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from typing import Tuple, Optional


class DepthEstimator:
    """
    Estimates sparse depth from epipolar geometry using Sampson distance
    """
    
    def __init__(self, alpha: float = 3.0, percentile: int = 95,rbf_sigma: float = 20.0):
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
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0,confidence=0.99)
        
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
    
    def estimate_sparse_depth(self, pts1: np.ndarray, pts2: np.ndarray,F: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate sparse depth weights from Sampson distance
        
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
        
        sampson_distances = self.compute_sampson_distance(pts1, pts2, F)

        d_max = np.percentile(sampson_distances, self.percentile)
        if d_max < 1e-6:
            d_max = sampson_distances.max() + 1e-6
        
        normalized_distances = np.clip(sampson_distances / d_max, 0, 1)
        
        # Convert to background weights
        # High distance = foreground = low weight
        # Low distance = background = high weight
        depth_weights = 1.0 / (1.0 + self.alpha * normalized_distances)
        
        outlier_mask = sampson_distances > 3 * d_max
        depth_weights[outlier_mask] = 0.1
        
        return depth_weights, F
    
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
        
        rbf = Rbf(x_coords, y_coords, values,function='gaussian',smooth=self.rbf_sigma)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        depth_map = rbf(grid_x, grid_y)
        depth_map = np.clip(depth_map, 0, 1)
        
        return depth_map.astype(np.float32)
    
    # def _interpolate_nearest(self, pts: np.ndarray, values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    #     """
    #     Nearest neighbor interpolation (fast but less smooth)
    #     """
    #     h, w = shape
        
    #     # Build KD-tree
    #     tree = cKDTree(pts)
        
    #     # Query for each pixel
    #     grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    #     grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
    #     distances, indices = tree.query(grid_pts, k=1)
        
    #     # Get values
    #     depth_map = values[indices].reshape(h, w)
        
    #     return depth_map.astype(np.float32)
    
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