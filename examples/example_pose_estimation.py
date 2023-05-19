import cv2
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modules.pose_estimation import PoseEstimator
from src.core.visualization import ImageVisualizer

def main():
    # Load camera calibration (example values - should be loaded from file)
    camera_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(camera_matrix, dist_coeffs)
    
    # Load template and image
    template_path = "../data/RS_img/cube_template.jpg"
    image_path = "../data/RS_img/cube_image.jpg"
    
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    
    if template is None or image is None:
        print("Error: Could not load images")
        return
    
    # Estimate pose
    try:
        cube_corners_2d, rvec, tvec, matches = pose_estimator.estimate_cube_pose(image, template)
        result_image = pose_estimator.draw_cube(image, cube_corners_2d)
        
        # Display results
        visualizer = ImageVisualizer()
        visualizer.display_images([
            ("Original Image", image),
            ("Cube Pose Estimation", result_image)
        ], layout=(1, 2))
        
        # Save result
        cv2.imwrite("pose_estimation_result.jpg", result_image)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()