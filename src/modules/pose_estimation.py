import cv2
import numpy as np
from src.core.feature_detection import ORBFeatureDetector

class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_detector = ORBFeatureDetector()
    
    def estimate_cube_pose(self, image, template, cube_size=0.1):
        """Estimate the pose of a cube in an image using solvePnP"""
        # Find keypoints and descriptors
        kp_template, des_template = self.feature_detector.detect_and_compute(template)
        kp_image, des_image = self.feature_detector.detect_and_compute(image)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_template, des_image)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_image[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Define 3D cube corners (object points)
        cube_corners_3d = np.float32([
            [0, 0, 0], 
            [cube_size, 0, 0], 
            [cube_size, cube_size, 0], 
            [0, cube_size, 0],
            [0, 0, -cube_size], 
            [cube_size, 0, -cube_size], 
            [cube_size, cube_size, -cube_size], 
            [0, cube_size, -cube_size]
        ])
        
        # Use solvePnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            cube_corners_3d, 
            dst_pts, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        if not success:
            raise RuntimeError("Pose estimation failed")
        
        # Project 3D points to 2D for drawing
        cube_corners_2d, _ = cv2.projectPoints(
            cube_corners_3d, 
            rvec, 
            tvec, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        return cube_corners_2d, rvec, tvec, matches
    
    def draw_cube(self, image, cube_corners_2d):
        """Draw a cube on the image based on projected points"""
        output = image.copy()
        cube_corners_2d = np.int32(cube_corners_2d).reshape(-1, 2)
        
        # Draw bottom face (front face)
        cv2.drawContours(output, [cube_corners_2d[:4]], -1, (0, 255, 0), 3)
        
        # Draw top face (back face)
        cv2.drawContours(output, [cube_corners_2d[4:]], -1, (0, 255, 0), 3)
        
        # Draw vertical edges
        for i in range(4):
            cv2.line(output, tuple(cube_corners_2d[i]), tuple(cube_corners_2d[i+4]), (0, 255, 0), 3)
        
        return output