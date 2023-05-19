import cv2
import numpy as np
import yaml

class Camera:
    def __init__(self, calibration_file=None):
        self.matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, file_path):
        """Load camera calibration parameters from YAML file"""
        try:
            with open(file_path, 'r') as file:
                calibration_data = yaml.safe_load(file)
                
            self.matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['distortion_coefficients'])
            self.calibrated = True
            print("Camera calibration loaded successfully")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self.calibrated = False
    
    def undistort(self, image):
        """Undistort image using camera calibration parameters"""
        if not self.calibrated:
            print("Camera not calibrated. Returning original image.")
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, self.matrix, self.dist_coeffs, None, new_camera_matrix
        )
        
        # Crop the image
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w]