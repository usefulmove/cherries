# Cherry Processing Operations

## Build & Test

### Building the Workspace
The system uses `colcon` for build management. Always run from the root of the workspace.

```bash
# Build entire workspace
colcon build --symlink-install

# Build specific package
colcon build --packages-select cherry_detection
```

### Running Tests
```bash
# Test entire workspace
colcon test

# Check test results
colcon test-result --all
```

## System Startup

### Standard Pipeline
To launch the full conveyor control system:

```bash
ros2 launch control_node conveyor_control_launch.py
```

### Robotic Sorter Integration
To launch the system with FANUC robot communication enabled:

```bash
ros2 launch fanuc_comms fanuc_launch.py
```

## Runtime Configuration

### Algorithm Selection
The `detection_server` can be configured at runtime to use different model variants:

```bash
# Get current algorithm
ros2 param get /detection_server algorithm

# Set algorithm (options: hdr_v1, hdr_v2, fasterRCNN-Mask_ResNet50_V1)
ros2 param set /detection_server algorithm fasterRCNN-Mask_ResNet50_V1
```

### Detection Thresholds
```bash
# Adjust pit detection confidence (0.0 - 1.0)
ros2 param set /detection_server pick_threshold 0.75

# Enable/Disable image saving for dataset collection
ros2 param set /control_node enable_image_save True
```

## Reference: Classification Codes
The system uses the following integer codes for cherry classification:

| Code | Label | Description |
|------|-------|-------------|
| 1 | Clean | Good cherry; allowed to pass. |
| 2 | Pit | Defective; targeted for rejection/sorting. |
| 3 | Side | Edge of belt; position unreliable. |
| 5 | Maybe | Uncertain; yellow visual feedback. |
