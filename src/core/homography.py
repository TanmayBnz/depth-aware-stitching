import cv2
import numpy as np
from typing import Tuple, Optional


class DepthAwareHomography:
    """
    Estimates homography with depth-weighted RANSAC
    IMPROVED: Better sampling strategy and diagnostics
    """
    
    def __init__(self, ransac_iterations: int = 2000, inlier_threshold: float = 3.0):
        """
        Initialize homography estimator
        
        Args:
            ransac_iterations: Number of RANSAC iterations
            inlier_threshold: Inlier threshold in pixels
        """
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold
        self.debug_info = {}  # Store debug information
    
    def estimate(self, pts1: np.ndarray, pts2: np.ndarray,
                 depth_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate homography
        
        Args:
            pts1, pts2: Point correspondences (n_points, 2)
            depth_weights: Optional depth weights (n_points,)
        
        Returns:
            H: Homography matrix (3, 3)
            inliers: Boolean mask of inliers (n_points,)
        """
        if len(pts1) < 4:
            raise ValueError("Need at least 4 point correspondences")
        
        if depth_weights is None:
            # standard
            return self._standard_ransac(pts1, pts2)
        else:
            # depth-weighted
            return self._depth_weighted_ransac(pts1, pts2, depth_weights)
    
    def _standard_ransac(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard homography estimation with RANSAC
        """
        H, mask = cv2.findHomography(
            pts1, pts2, 
            cv2.RANSAC, 
            self.inlier_threshold
        )
        
        if H is None:
            H, _ = cv2.findHomography(pts1, pts2, 0)
            mask = np.ones(len(pts1), dtype=np.uint8)
        
        inliers = mask.ravel().astype(bool)
        
        return H, inliers
    
    def _depth_weighted_ransac(self, pts1: np.ndarray, pts2: np.ndarray,
                               depth_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMPROVED Depth-weighted RANSAC with better sampling strategy
        """
        n_points = len(pts1)
        best_H = None
        best_score = -1
        best_inliers = None
        
        # IMPROVED: Use less aggressive weighting for sampling
        # Square root makes distribution less extreme
        sample_probs = np.sqrt(depth_weights)
        sample_probs = sample_probs / (sample_probs.sum() + 1e-8)
        
        # Ensure minimum probability for all points
        min_prob = 0.02  # Increased from 0.01
        sample_probs = np.maximum(sample_probs, min_prob)
        sample_probs = sample_probs / sample_probs.sum()
        
        # Track statistics
        bg_samples = 0
        fg_samples = 0
        is_background = depth_weights > 0.7
        
        for iteration in range(self.ransac_iterations):
            try:
                sample_indices = np.random.choice(n_points, size=4, replace=False, p=sample_probs)
            except:
                sample_indices = np.random.choice(n_points, size=4, replace=False)
            
            # Track sampling statistics
            bg_samples += is_background[sample_indices].sum()
            fg_samples += (~is_background[sample_indices]).sum()
            
            sample_pts1 = pts1[sample_indices]
            sample_pts2 = pts2[sample_indices]

            H = self._compute_homography_dlt(sample_pts1, sample_pts2)
            
            if H is None:
                continue
            
            # Compute reprojection errors
            errors = self._compute_reprojection_errors(pts1, pts2, H)
            inliers = errors < self.inlier_threshold
            
            # IMPROVED: Use both weighted score AND inlier count
            # This prevents selecting models with few but high-weight inliers
            weighted_score = (inliers * depth_weights).sum()
            inlier_count = inliers.sum()
            
            # Combined score: weight * sqrt(count) to balance quality and quantity
            combined_score = weighted_score * np.sqrt(inlier_count)
            
            if combined_score > best_score:
                best_score = combined_score
                best_H = H
                best_inliers = inliers
        
        # Store debug info
        self.debug_info = {
            'bg_sample_ratio': bg_samples / (bg_samples + fg_samples),
            'fg_sample_ratio': fg_samples / (bg_samples + fg_samples),
            'expected_bg_ratio': is_background.mean(),
            'best_score': best_score
        }
        
        if best_H is None:
            print("  [Warning] Depth-weighted RANSAC failed, falling back to standard")
            return self._standard_ransac(pts1, pts2)
        
        # Refine using weighted least squares on inliers
        inlier_pts1 = pts1[best_inliers]
        inlier_pts2 = pts2[best_inliers]
        inlier_weights = depth_weights[best_inliers]
        
        refined_H = self._weighted_least_squares(inlier_pts1, inlier_pts2, inlier_weights)
        
        # Recompute inliers with refined H
        errors = self._compute_reprojection_errors(pts1, pts2, refined_H)
        final_inliers = errors < self.inlier_threshold
        
        return refined_H, final_inliers
    
    def _compute_homography_dlt(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography using Direct Linear Transform (DLT)
        
        Args:
            pts1, pts2: 4 point correspondences
        
        Returns:
            H: Homography matrix (3, 3) or None if computation fails
        """
        if len(pts1) < 4:
            return None
        
        A = []
        for i in range(len(pts1)):
            x, y = pts1[i]
            x_prime, y_prime = pts2[i]
            
            A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
        
        A = np.array(A)
        
        try:
            U, S, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            
            H = H / H[2, 2]
            
            return H
        except:
            return None
    
    def _weighted_least_squares(self, pts1: np.ndarray, pts2: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute homography using weighted least squares
        
        This gives more importance to background points in refinement
        """
        n = len(pts1)
        A = []
        
        for i in range(n):
            x, y = pts1[i]
            x_prime, y_prime = pts2[i]
            w = np.sqrt(weights[i])
            
            # Weighted equations
            A.append(w * np.array([
                -x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime
            ]))
            A.append(w * np.array([
                0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime
            ]))
        
        A = np.array(A)
        
        # Solve using SVD
        try:
            U, S, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            H = H / H[2, 2]
            return H
        except:
            # Fallback to unweighted
            return self._compute_homography_dlt(pts1, pts2)
    
    def _compute_reprojection_errors(self, pts1: np.ndarray, pts2: np.ndarray,
                                    H: np.ndarray) -> np.ndarray:
        """
        Compute reprojection error for each point
        
        Error = || pts2 - H * pts1 ||
        """
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        
        pts1_projected = (H @ pts1_h.T).T
        
        # Convert back to Cartesian
        pts1_projected = pts1_projected[:, :2] / (pts1_projected[:, 2:3] + 1e-8)
        
        # Compute Euclidean distance
        errors = np.linalg.norm(pts2 - pts1_projected, axis=1)
        
        return errors
    
    def get_inlier_statistics(self, pts1: np.ndarray, pts2: np.ndarray,
                              H: np.ndarray, inliers: np.ndarray,
                              depth_weights: Optional[np.ndarray] = None) -> dict:
        """
        IMPROVED: Compute comprehensive statistics about the homography and inliers
        """
        errors = self._compute_reprojection_errors(pts1, pts2, H)
        inlier_errors = errors[inliers]
        
        stats = {
            'n_inliers': inliers.sum(),
            'inlier_ratio': inliers.sum() / len(inliers),
            'mean_inlier_error': inlier_errors.mean() if len(inlier_errors) > 0 else 0,
            'std_inlier_error': inlier_errors.std() if len(inlier_errors) > 0 else 0,
            'median_inlier_error': np.median(inlier_errors) if len(inlier_errors) > 0 else 0
        }
        
        # Add depth-aware statistics if weights provided
        if depth_weights is not None:
            # Background/foreground breakdown
            is_background = depth_weights > 0.7
            is_foreground = depth_weights < 0.4
            
            bg_inliers = inliers & is_background
            fg_inliers = inliers & is_foreground
            
            stats['bg_inliers'] = bg_inliers.sum()
            stats['fg_inliers'] = fg_inliers.sum()
            stats['bg_inlier_ratio'] = bg_inliers.sum() / max(is_background.sum(), 1)
            stats['fg_inlier_ratio'] = fg_inliers.sum() / max(is_foreground.sum(), 1)
            
            # Weighted inlier score
            stats['weighted_inlier_score'] = (inliers * depth_weights).sum()
            
            # Average depth of inliers
            if inliers.sum() > 0:
                stats['mean_inlier_depth_weight'] = depth_weights[inliers].mean()
            else:
                stats['mean_inlier_depth_weight'] = 0
        
        return stats
    
    def get_debug_info(self) -> dict:
        """Get debug information from last weighted RANSAC run"""
        return self.debug_info