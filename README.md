# Computer Vision Projects

A modular and well-structured computer vision project with OOP design and clear separation of concerns.

## Repository Structure

```
computer-vision-projects/
├── README.md
├── requirements.txt
├── config/
│   └── camera_calibration.yaml
├── data/
│   ├── test_images_rr/
│   └── RS_img/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── feature_detection.py
│   │   └── visualization.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── corner_detection.py
│   │   ├── pose_estimation.py
│   │   └── stereo_vision.py
└── examples/
    ├── example_corner_detection.py
    ├── example_pose_estimation.py
    └── example_stereo_vision.py
```

## Key Features

- **Modular Design**: Each functionality is separated into its own module with clear responsibilities
- **OOP Principles**: Used inheritance, abstraction, and encapsulation throughout
- **Configuration Management**: Centralized camera calibration and parameters
- **Error Handling**: Added proper exception handling and validation
- **Code Reusability**: Created base classes and utility functions for common operations
- **Documentation**: Added docstrings and comments for better understanding

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your calibration data in `config/camera_calibration.yaml` (or use example values in code for testing)

3. Add your images to the `data/` directory

## Usage Examples

### Corner Detection
```bash
python examples/example_corner_detection.py
```

### Cube Pose Estimation
```bash
python examples/example_pose_estimation.py
```

### Stereo Vision
```bash
python examples/example_stereo_vision.py
```

## Requirements

- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- PyYAML (`pyyaml`)
- Matplotlib (`matplotlib`)

## Notes

- For stereo vision, you need stereo calibration parameters (R, T) between the cameras. These should be obtained from a stereo calibration process.
- The `camera_calibration.yaml` file should contain parameters for single camera calibration. For stereo, you'll need a separate stereo calibration file.
- The `data/` directory should contain test images for the examples.

