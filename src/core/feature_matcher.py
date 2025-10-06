"""
feature detection using SIFT with lowe's ratio
"""

import cv2
import numpy as np
from typing import Tuple, List
class FeatureMatcher:
    """
    handles feature detection and matching
    """
    def __init__(self, nfeatures:int = 5000, ratio_threshold: float = 0.75):
        """
        initialise feature matcher
        """
        self.nfeatures = nfeatures
        self.ratio_threshold = ratio_threshold
        self.detector = cv2.SIFT_create(nfeatures = nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        detect keypoints and compute descriptors
        arg: input image
        returns:
            keypoints
            descriptors
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            return [], np.array([])
        return keypoints, descriptors
    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        match using lowe's ratio test
        """
        if len(des1) == 0 or len(des2) == 0:
            return []
        matches = self.matcher.knnMatch(des1, des2, k =2)
        good_matches = []
        for match_pair in matches:
            m, n  = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def extract_matched_points(self, k1:List, k2:List, matches:List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrat point coords from matches
        """
        if len(matches) == 0:
            return np.array([]), np.array([])
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
        return pts1, pts2
    
    def visualize_matches(self,img1: np.ndarray,kp1: List,img2: np.ndarray,kp2: List,matches: List[cv2.DMatch],max_display: int = 50) -> np.ndarray:
        """
        Visualize feature matches
        """
        matches_subset = matches[:max_display] if len(matches) > max_display else matches

        vis_image = cv2.drawMatches(
            img1, kp1, img2, kp2, matches_subset, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return vis_image
    
    def get_match_statistics(self, keypoints1: List, keypoints2: List, matches: List[cv2.DMatch]) -> dict:
        """
        Compute statistics about matches
        """
        stats = {
            'n_keypoints_1': len(keypoints1),
            'n_keypoints_2': len(keypoints2),
            'n_matches': len(matches),
            'match_ratio': len(matches) / max(len(keypoints1), 1)
        }
        
        if len(matches) > 0:
            distances = [m.distance for m in matches]
            stats['mean_distance'] = np.mean(distances)
            stats['std_distance'] = np.std(distances)
            stats['min_distance'] = np.min(distances)
            stats['max_distance'] = np.max(distances)
        
        return stats