import cv2
import numpy as np
from abc import ABC, abstractmethod

class CornerDetector(ABC):
    """Abstract base class for corner detectors"""
    
    @abstractmethod
    def detect(self, image):
        pass
    
    @abstractmethod
    def draw_corners(self, image, corners):
        pass


class HarrisCornerDetector(CornerDetector):
    def __init__(self, block_size=2, ksize=3, k=0.04, threshold_ratio=0.17):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold_ratio = threshold_ratio
    
    def detect(self, image):
        """Detect corners using Harris method"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        gray = np.float32(gray)
        corners = cv2.cornerHarris(gray, self.block_size, self.ksize, self.k)
        corners = cv2.dilate(corners, None)
        
        return corners
    
    def get_keypoints(self, corners):
        """Convert corner response to keypoints"""
        threshold = corners.max() * self.threshold_ratio
        keypoints = []
        
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                if corners[i, j] > threshold:
                    keypoints.append(cv2.KeyPoint(j, i, 5))
        
        return keypoints
    
    def draw_corners(self, image, corners, color=(0, 255, 0), radius=5):
        """Draw detected corners on image"""
        output = image.copy()
        threshold = corners.max() * self.threshold_ratio
        
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                if corners[i, j] > threshold:
                    cv2.circle(output, (j, i), radius, color, -1)
        
        return output


class SubPixelCornerDetector(HarrisCornerDetector):
    def __init__(self, block_size=2, ksize=3, k=0.04, threshold_ratio=0.01, 
                 subpix_window=(5, 5), subpix_zero_zone=(-1, -1), criteria_max_iter=100):
        super().__init__(block_size, ksize, k, threshold_ratio)
        self.subpix_window = subpix_window
        self.subpix_zero_zone = subpix_zero_zone
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                         criteria_max_iter, 0.001)
    
    def detect(self, image):
        """Detect corners with subpixel refinement"""
        corners = super().detect(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Threshold and find centroids
        _, thresh = cv2.threshold(corners, self.threshold_ratio * corners.max(), 255, 0)
        thresh = np.uint8(thresh)
        
        # Find connected components
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        # Refine corners
        refined_corners = cv2.cornerSubPix(
            gray, np.float32(centroids), 
            self.subpix_window, self.subpix_zero_zone, self.criteria
        )
        
        return refined_corners, centroids
    
    def draw_corners(self, image, refined_corners, original_centroids, 
                    original_color=(0, 0, 255), refined_color=(0, 255, 0)):
        """Draw both original and refined corners"""
        output = image.copy()
        refined_corners = np.int0(refined_corners)
        original_centroids = np.int0(original_centroids)
        
        # Draw original centroids in red
        output[original_centroids[:, 1], original_centroids[:, 0]] = original_color
        
        # Draw refined corners in green
        output[refined_corners[:, 1], refined_corners[:, 0]] = refined_color
        
        return output