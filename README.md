# Traffic Signal Detection

This project implements a traffic signal detection system using ROS and OpenCV. It can detect crosswalks and traffic lights in real-time video streams.

## Features

- Crosswalk detection
- Traffic light detection (red, yellow, green)
- Real-time visualization
- ROS integration
- Test framework with video support

## Requirements

- ROS Noetic
- OpenCV
- Python 3
- cv_bridge
- image_transport

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-signal-detection.git
cd traffic-signal-detection
```

2. Build the package:
```bash
catkin_make
```

3. Source the workspace:
```bash
source devel/setup.bash
```

## Usage

1. Start the detector node:
```bash
rosrun detector_try detector_node
```

2. Run the test script:
```bash
cd src/detector_try
python3 scripts/test_detector_node.py
```

## Project Structure

```
detector_try/
├── config/             # Configuration files
├── include/            # Header files
├── launch/            # Launch files
├── rviz/              # RViz configuration
├── scripts/           # Python scripts
├── src/               # Source files
├── test_results/      # Test results
└── test_videos/       # Test videos
```

## Testing

The project includes a test framework that can:
- Process test videos
- Generate test reports
- Visualize detection results
- Calculate detection accuracy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 