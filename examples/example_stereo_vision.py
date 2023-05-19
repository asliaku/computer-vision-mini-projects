import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modules.stereo_vision import StereoProcessor
from src.core.visualization import ImageVisualizer

def main():
    # Load stereo calibration parameters (example values)
    # These should come from a calibration file
    camera_matrix_left = np.array([[800, 0, 320],
                                  [0, 800, 240],
                                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs_left = np.zeros((4, 1))
    
    camera_matrix_right = np.array([[800, 0, 320],
                                   [0, 800, 240],
                                   [0, 0, 1]], dtype=np.float32)
    dist_coeffs_right = np.zeros((4, 1))
    
    # Rotation and translation between cameras (example)
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.1, 0, 0], dtype=np.float32)
    
    image_size = (640, 480)  # Example image size
    
    # Initialize stereo processor
    stereo_processor = StereoProcessor(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        R, T, image_size
    )
    
    # Load stereo images
    left_img_path = "../data/RS_img/left_image.jpg"
    right_img_path = "../data/RS_img/right_image.jpg"
    
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print("Error: Could not load stereo images")
        return
    
    # Rectify images
    left_rect, right_rect = stereo_processor.rectify_images(left_img, right_img)
    
    # Compute disparity
    disparity = stereo_processor.compute_disparity(left_rect, right_rect)
    
    # Compute depth map
    depth_map = stereo_processor.compute_depth_map(disparity)
    
    # Display results
    visualizer = ImageVisualizer(figsize=(20, 10))
    visualizer.display_images([
        ("Left Image", left_img),
        ("Right Image", right_img),
        ("Rectified Left", left_rect),
        ("Rectified Right", right_rect),
        ("Disparity Map", disparity),
        ("Depth Map", depth_map)
    ], layout=(2, 3))
    
    # Save results
    cv2.imwrite("rectified_left.jpg", left_rect)
    cv2.imwrite("rectified_right.jpg", right_rect)
    cv2.imwrite("disparity.jpg", disparity / disparity.max() * 255)
    cv2.imwrite("depth_map.jpg", depth_map)

if __name__ == "__main__":
    main()