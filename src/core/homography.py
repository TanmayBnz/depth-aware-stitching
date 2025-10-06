import cv2
import numpy as np
from typing import Tuple, Optional


class DepthAwareHomography:
    """
    Estimates homography with depth-weighted RANSAC
    """
    
    def __init__(self, ransac_iterations: int = 2000,inlier_threshold: float = 3.0):
        """
        Initialize homography estimator
        
        Args:
            ransac_iterations: Number of RANSAC iterations
            inlier_threshold: Inlier threshold in pixels
        """
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold
    
    def estimate(self, pts1: np.ndarray, pts2: np.ndarray,depth_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _depth_weighted_ransac(self, pts1: np.ndarray, pts2: np.ndarray,depth_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Depth-weighted RANSAC
        
        innovation: Sample points with probability proportional to depth weight
        """
        n_points = len(pts1)
        best_H = None
        best_score = -1
        best_inliers = None
        
        sample_probs = depth_weights / (depth_weights.sum() + 1e-8)
        
        sample_probs = np.maximum(sample_probs, 0.01)
        sample_probs = sample_probs / sample_probs.sum()
        
        for iteration in range(self.ransac_iterations):
            try:
                sample_indices = np.random.choice(n_points, size=4, replace=False, p=sample_probs)
            except:
                sample_indices = np.random.choice(n_points, size=4, replace=False)
            
            sample_pts1 = pts1[sample_indices]
            sample_pts2 = pts2[sample_indices]

            H = self._compute_homography_dlt(sample_pts1, sample_pts2)
            
            if H is None:
                continue
            
            # Compute reprojection
            errors = self._compute_reprojection_errors(pts1, pts2, H)

            inliers = errors < self.inlier_threshold
            
            # Compute WEIGHTED score
            weighted_score = (inliers * depth_weights).sum()
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_H = H
                best_inliers = inliers
        
        if best_H is None:
            return self._standard_ransac(pts1, pts2)
        
        # refine using weighted least squares on inliers
        inlier_pts1 = pts1[best_inliers]
        inlier_pts2 = pts2[best_inliers]
        inlier_weights = depth_weights[best_inliers]
        
        refined_H = self._weighted_least_squares(inlier_pts1, inlier_pts2, inlier_weights)
        
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
    
    def _compute_reprojection_errors(self, pts1: np.ndarray, pts2: np.ndarray,H: np.ndarray) -> np.ndarray:
        """
        Compute reprojection error for each point
        
        Error = || pts2 - H * pts1 ||
        """
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        
        pts1_projected = (H @ pts1_h.T).T
        
        # Convert back to Cartesian
        pts1_projected = pts1_projected[:, :2] / pts1_projected[:, 2:3]
        
        # Compute Euclidean distance
        errors = np.linalg.norm(pts2 - pts1_projected, axis=1)
        
        return errors
    
    def get_inlier_statistics(self, pts1: np.ndarray, pts2: np.ndarray,H: np.ndarray,inliers: np.ndarray) -> dict:
        """
        Compute statistics about the homography and inliers
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
        
        return stats