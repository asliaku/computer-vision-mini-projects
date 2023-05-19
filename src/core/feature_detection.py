import cv2
import numpy as np

class ORBFeatureDetector:
    def __init__(self, nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, 
                 firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, 
                 fastThreshold=20):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold
        )
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors for the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors