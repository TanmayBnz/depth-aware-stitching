import cv2
import numpy as np
from typing import Tuple, Optional


class DepthAwareHomography:
    """
    Estimates homography with depth-weighted RANSAC
    IMPROVED: Automatically selects the best model between
              standard RANSAC and depth-weighted RANSAC based
              on a robust, un-weighted inlier count.
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
        Estimate homography with an internal "safety net".
        
        If depth_weights are not provided, this runs standard RANSAC.
        
        If depth_weights ARE provided, this will:
        1. Compute a standard RANSAC homography (H_std) and count its inliers.
        2. Compute a depth-weighted RANSAC homography (H_dw) and count its inliers.
        3. Return the homography that found the *most inliers*,
           breaking ties in favor of the depth-weighted one.
        
        This provides an inherent "safety net" against "poisoned" depth
        weights. The depth-aware method must prove it's better by
        finding a larger consensus set (i.e., the true background plane).
        
        Args:
            pts1, pts2: Point correspondences (n_points, 2)
            depth_weights: Optional depth weights (n_points,)
        
        Returns:
            H: The best homography matrix (3, 3)
            inliers: Boolean mask of inliers for the best H (n_points,)
        """
        if len(pts1) < 4:
            raise ValueError("Need at least 4 point correspondences")
        
        if depth_weights is None:
            # No weights provided, run standard RANSAC
            self.debug_info['final_choice'] = 'standard (no weights)'
            return self._standard_ransac(pts1, pts2)
        else:
            # --- START HYBRID "SAFETY NET" LOGIC ---
            
            # 1. Get the Standard RANSAC candidate
            H_std, inliers_std_mask = self._standard_ransac(pts1, pts2)
            if H_std is None: # Should only happen on total failure
                H_std, _ = cv2.findHomography(pts1, pts2, 0) # Fallback to DLT
            
            # 2. Get the Depth-Weighted candidate
            # We only care about the *hypothesis* (H_dw), not its inliers
            H_dw, _ = self._depth_weighted_ransac(pts1, pts2, depth_weights)

            if H_dw is None:
                # Depth-weighted failed, must use standard
                self.debug_info['final_choice'] = 'standard (dw_failed)'
                return H_std, inliers_std_mask

            # 3. Objectively score both candidates
            # We re-calculate inliers for both H using a simple, un-weighted check.
            # The model that finds the *largest consensus set* (most inliers) wins.
            
            errors_std = self._compute_reprojection_errors(pts1, pts2, H_std)
            inliers_mask_std = errors_std < self.inlier_threshold
            inlier_count_std = np.sum(inliers_mask_std)

            errors_dw = self._compute_reprojection_errors(pts1, pts2, H_dw)
            inliers_mask_dw = errors_dw < self.inlier_threshold
            inlier_count_dw = np.sum(inliers_mask_dw)

            # 4. Compare and return the winner
            if inlier_count_dw >= inlier_count_std:
                # Depth-weighted found a larger or equal consensus set.
                # This means it found the true background plane.
                self.debug_info['final_choice'] = 'depth_weighted'
                
                # Refine using weighted least squares on *its* inliers
                refined_H = self._weighted_least_squares(pts1[inliers_mask_dw], pts2[inliers_mask_dw], depth_weights[inliers_mask_dw])
                final_errors = self._compute_reprojection_errors(pts1, pts2, refined_H)
                final_inliers = final_errors < self.inlier_threshold
                
                return refined_H, final_inliers
            
            else:
                # Standard RANSAC found a larger consensus set.
                # This means the depth weights were "poisoned". Fall back to standard.
                self.debug_info['final_choice'] = 'standard (fallback)'

                # Refine using weighted least squares on *its* inliers
                refined_H = self._weighted_least_squares(pts1[inliers_mask_std], pts2[inliers_mask_std], depth_weights[inliers_mask_std])
                final_errors = self._compute_reprojection_errors(pts1, pts2, refined_H)
                final_inliers = final_errors < self.inlier_threshold

                return refined_H, final_inliers
            # --- END HYBRID "SAFETY NET" LOGIC ---

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
            # RANSAC failed, fallback to DLT on all points
            H, _ = cv2.findHomography(pts1, pts2, 0)
            if H is None:
                return None, np.zeros(len(pts1), dtype=bool)
            mask = np.ones(len(pts1), dtype=np.uint8)
        
        inliers = mask.ravel().astype(bool)
        
        return H, inliers
    
    def _depth_weighted_ransac(self, pts1: np.ndarray, pts2: np.ndarray,
                               depth_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMPROVED Depth-weighted RANSAC with better sampling strategy
        Returns: H, inliers
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
        min_prob = 0.02
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
            'bg_sample_ratio': bg_samples / (bg_samples + fg_samples + 1e-8),
            'fg_sample_ratio': fg_samples / (bg_samples + fg_samples + 1e-8),
            'expected_bg_ratio': is_background.mean(),
            'best_dw_score': best_score
        }
        
        if best_H is None:
            # Depth-weighted RANSAC found no model
            return None, np.zeros(n_points, dtype=bool)
        
        return best_H, best_inliers
    
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
        if n < 4:
            # Not enough points, return DLT on all available
            H_dlt = self._compute_homography_dlt(pts1, pts2)
            if H_dlt is None: # Total failure, return identity
                 return np.identity(3)
            return H_dlt

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
        if H is None:
             return { 'n_inliers': 0, 'inlier_ratio': 0, 'mean_inlier_error': float('inf') }

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