import cv2
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modules.corner_detection import HarrisCornerDetector, SubPixelCornerDetector
from src.core.visualization import ImageVisualizer

def main():
    # Initialize detectors
    harris_detector = HarrisCornerDetector(threshold_ratio=0.17)
    subpix_detector = SubPixelCornerDetector(threshold_ratio=0.01)
    
    # Load image
    image_path = "../data/test_images_rr/infrared_frame2_1.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect corners with Harris
    corners = harris_detector.detect(image)
    harris_result = harris_detector.draw_corners(image, corners)
    
    # Detect corners with subpixel refinement
    refined_corners, centroids = subpix_detector.detect(image)
    subpix_result = subpix_detector.draw_corners(image, refined_corners, centroids)
    
    # Display results
    visualizer = ImageVisualizer()
    visualizer.display_images(
        [("Original", image), ("Harris Corners", harris_result), ("Subpixel Refined", subpix_result)],
        layout=(1, 3)
    )

if __name__ == "__main__":
    main()