# Cherry Processing Machine - Development & Operations Guide

## File System Structure

**Main Working Directory:**
```
Extreme SSD/traina cherry line/
```

**ROS2 Workspace Structure:**
```
cherry_ws_ws/src/cherry_system/
├── cherry_detection/          # AI detection service
├── control_node/              # System orchestration
├── tracking_projector/        # Visual feedback system
├── camera_simulator/          # Testing/simulation tool
├── usb_io/                    # Hardware I/O interface
├── fanuc_comms/               # Robotic communication
├── avt_vimba_camera/          # Allied Vision camera interface
└── cherry_interfaces/         # Custom messages/services
```

**Key Development Files:**
```
backup/cherry_ws_ws/src/cherry_system/control_node/control_node/control_node.py
backup/cherry_ws_ws/src/cherry_system/cherry_detection/cherry_detection/ai_detector.py
backup/cherry_ws_ws/src/cherry_system/cherry_interfaces/msg/Cherry.msg
```

**Data Storage Locations:**
```
/home/user/Pictures/hdr/              # Image collection
cpytorch/                             # ML development
Downloads/                            # Training notebooks
```

## Development Commands

**Build Commands:**
```bash
# Build entire workspace
colcon build --symlink-install

# Build specific package
colcon build --packages-select <pkg>
```

**Test Commands:**
```bash
# Test entire workspace
colcon test

# Test specific package
colcon test --packages-select <pkg>

# Test specific functionality
colcon test --packages-select <pkg> --pytest-args -k <test_name>

# Check test results
colcon test-result --all
```

**Safe Mode Development:**
```bash
# Enable simulation mode (no hardware required)
ros2 param set /usb_io_node simulate_encoder True

# Use camera simulator instead of real cameras
ros2 launch camera_simulator camera_simulator_launch.py
```

## Code Style Guidelines

**Python Requirements:**
- **PEP8 compliant** - Use snake_case for functions/variables, PascalCase for classes
- **Import organization**: Group by (1) standard lib, (2) ROS2 modules, (3) local packages
- **Error handling**: Wrap hardware/service calls in try-except blocks
- **Logging**: Use `self.get_logger().error()` for error reporting

**C++ Requirements:**
- **PascalCase** for classes and methods
- **Private members** end with underscore (e.g., `nh_`)
- **Error handling**: Use try-catch blocks with `RCLCPP_ERROR()` for logging

**Framework Usage:**
- **ROS2**: Use `rclpy` (Python) or `rclcpp` (C++) with custom interfaces from `cherry_interfaces`
- **Naming**: Use descriptive names for nodes and topics (e.g., `detector_node`, `~/detect`)
- **Coordinates**: Convert pixels to real-world meters using `Frame_tf` before tracking/actuation

## Data Management

**Image Collection:**
```bash
# Enable image saving
ros2 param set /image_services enable_saving True

# Set save location (default: /home/user/Pictures/hdr/)
ros2 param set /image_services base_path /custom/path/
```

**Training Datasets:**
- Multiple versions in `pytorch/` and `Downloads/`
- COCO format: Cherry Inspection v3
- YOLOv7 format: Cherry Inspection #2 v3
- Roboflow-managed datasets (CC BY 4.0 license)

**Model Files:**
```
classification-2_26_2025-iter5.pt    # Latest classification
classification-iter10.pt              # Earlier iteration
seg_model_red_v1.pt                   # Segmentation variant
NMS-nomask-1-3-2024.pt               # NMS variant
newlights_nomask_12-15-2023.pt       # Different lighting
```

## System Configuration

**Network Configuration:**
- PC IP: 172.16.1.223/16
- Camera IP: 172.16.2.2/16

**Runtime Parameters:**
```bash
# Algorithm selection
ros2 param get/set /detection_server algorithm <name>
# Options: hdr_v1, hdr_v2, vote_v1, fasterRCNN-Mask_ResNet50_V1

# Detection thresholds
ros2 param set /detection_server pick_threshold 0.5
ros2 param set /detection_server maybe_threshold <value>
```

**Startup Procedure:**
```bash
# Terminal 1: Launch main system
ros2 launch fanuc_comms fanuc_launch.py

# Terminal 2: Set detection algorithm
ros2 param set /detection_server algorithm fasterRCNN-Mask_ResNet50_V1
```

## System Status & Backups

**Production Status:** ✅ **OPERATIONAL & PRODUCTION-READY**
- Detection pipeline: Fully functional
- Hardware integration: Complete
- Real-time tracking: Implemented
- Visual feedback: Operational

**Backup Locations:**
```
cherry_ws_backup_4_29_2023/
cherry_backup_12_21_2022/
cherry_ws_backup_6_29_2023/
```

**Model Versioning:**
- Iterative improvements: iter5, iter10, etc.
- Multiple lighting condition variants
- Ensemble voting options (vote_v1)

## Development Notes

**System Strengths:**
1. Two-stage approach provides robustness
2. HDR imaging handles varying appearances
3. GPU acceleration enables real-time performance
4. Modular ROS2 architecture for scalability
5. Multiple model versions for A/B testing

**Technical Sophistication:**
- Coordinate transformation for accurate positioning
- Encoder integration for precise tracking
- Morphological operations for mask refinement
- Confidence-based classification
- Real-time projection with latency compensation

**Areas for Enhancement:**
- Comprehensive system documentation
- Automated testing framework
- Performance metrics logging
- Unified training pipeline documentation
