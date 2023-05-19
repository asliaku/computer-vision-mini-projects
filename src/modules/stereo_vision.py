import cv2
import numpy as np
from src.core.feature_detection import ORBFeatureDetector

class StereoProcessor:
    def __init__(self, camera_matrix_left, dist_coeffs_left, 
                 camera_matrix_right, dist_coeffs_right,
                 R, T, image_size):
        """
        Initialize stereo processor with calibration parameters.
        
        Parameters:
        - camera_matrix_left/right: Camera matrices for left and right cameras
        - dist_coeffs_left/right: Distortion coefficients
        - R: Rotation matrix between cameras
        - T: Translation vector between cameras
        - image_size: (width, height) of the images
        """
        self.cam_left = (camera_matrix_left, dist_coeffs_left)
        self.cam_right = (camera_matrix_right, dist_coeffs_right)
        self.R = R
        self.T = T
        self.image_size = image_size
        self.feature_detector = ORBFeatureDetector()
        self.stereo_matcher = None
        self.rectify_maps = None
        self.Q = None
        self.rectify()
    
    def rectify(self):
        """Compute rectification transforms and create rectification maps"""
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.cam_left[0], self.cam_left[1],
            self.cam_right[0], self.cam_right[1],
            self.image_size, self.R, self.T
        )
        
        # Create rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.cam_left[0], self.cam_left[1], R1, P1, 
            self.image_size, cv2.CV_32FC1
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.cam_right[0], self.cam_right[1], R2, P2,
            self.image_size, cv2.CV_32FC1
        )
        
        self.rectify_maps = {
            'left': (left_map1, left_map2),
            'right': (right_map1, right_map2)
        }
        self.Q = Q  # For depth map conversion
    
    def rectify_images(self, left_img, right_img):
        """Rectify stereo images using precomputed maps"""
        if self.rectify_maps is None:
            self.rectify()
        
        # Apply rectification
        left_rect = cv2.remap(
            left_img, 
            self.rectify_maps['left'][0], 
            self.rectify_maps['left'][1], 
            cv2.INTER_LINEAR
        )
        right_rect = cv2.remap(
            right_img, 
            self.rectify_maps['right'][0], 
            self.rectify_maps['right'][1], 
            cv2.INTER_LINEAR
        )
        
        return left_rect, right_rect
    
    def init_stereo_matcher(self, block_size=5, min_disp=0, num_disp=16):
        """Initialize stereo matcher with parameters"""
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    
    def compute_disparity(self, left_img, right_img):
        """Compute disparity map from stereo images"""
        if self.stereo_matcher is None:
            self.init_stereo_matcher()
        
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        return disparity
    
    def compute_depth_map(self, disparity):
        """Convert disparity map to depth map using Q matrix from stereoRectify"""
        # If we have Q matrix, use it for depth calculation
        if self.Q is not None:
            # Create a 3D point cloud from disparity
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
            # Depth is the Z-coordinate
            depth_map = points_3d[:, :, 2]
            # Normalize for visualization
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map = np.uint8(depth_map)
            return depth_map
        else:
            # Fallback to simple calculation
            disparity = disparity.astype(np.float32)
            disparity[disparity == 0] = 0.1
            depth_map = 1.0 / disparity
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            return np.uint8(depth_map)
    
    def find_stereo_correspondences(self, left_img, right_img):
        """Find matching features between stereo images"""
        # First rectify the images
        left_rect, right_rect = self.rectify_images(left_img, right_img)
        
        # Detect and compute features
        kp_left, des_left = self.feature_detector.detect_and_compute(left_rect)
        kp_right, des_right = self.feature_detector.detect_and_compute(right_rect)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_left, des_right)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return kp_left, kp_right, matches, left_rect, right_rect