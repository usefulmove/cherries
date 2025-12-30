# Cherry Processing Machine - Implementation Details

## ML Pipeline Architecture

**Two-Stage Detection Pipeline:**

**Stage 1 - Instance Segmentation:**
- Model: Mask R-CNN with ResNet50-FPN backbone
- Framework: PyTorch + torchvision
- Purpose: Detect and segment individual cherries in image
- File: `cherry_segmentation.pt`
- Outputs: Bounding boxes, masks, detection scores
- Parameters: RPN NMS threshold 0.7, box detection threshold 0.5, up to 1000 detections

**Stage 2 - Classification:**
- Model: ResNet50 with custom final layer
- Framework: PyTorch  
- Purpose: Classify each segmented cherry
- File: `cherry_classification.pt`
- Classes: 0 (Clean), 1 (Pit)
- Classification Logic:
  - **Pit**: confidence > 75% for pit class
  - **Clean**: confidence > 50% for clean class  
  - **Maybe**: confidence 50-75% for pit class
  - **Side**: Spatial filtering for edge-positioned cherries

**Image Processing Pipeline:**
1. Crop to relevant region: [200:700, 0:2463] pixels
2. HDR imaging with multiple exposures (bot1, bot2, top2)
3. Morphological operations (5x5 kernel erosion/dilation)
4. 128x128 patches extracted for classification

## Coordinate Transformation Pipeline

**Pixel-to-Real-World Conversion:**
- Scaling factor: 2710.316 pixels/mm
- Rotation transformation: π/2 radians rotation matrix
- Origin offset: (2448, 652, π/2)
- Implementation: `detector_node.py` and `frame_tf.py`

**Frame Tracking System:**
- `FrameTracker` class maintains cherry state across conveyor movement
- Updates positions based on encoder distance changes
- Handles cherry entry/exit from tracking window
- Publishes `CherryArrayStamped` messages for visualization

## ROS2 Service Architecture

**Core Services/Actions:**
- **`detection_server/detect`**: Main inference service
  - Input: Combined HDR image
  - Output: `CherryArray` with world coordinates and classifications
- **`Acquisition` action**: Coordinates camera capture
  - Triggered by control_node based on encoder position
  - Returns `ImageSet` with top/bottom camera images
- **`~/encoder_change` topic**: Conveyor position updates
  - Published by usb_io_node
  - Triggers new frame acquisition every ~500 ticks

**Message Types** (defined in cherry_interfaces):
- `Cherry.msg`: Individual cherry data (position, classification, confidence)
- `CherryArray.msg`: Array of detected cherries
- `CherryArrayStamped.msg`: Timed cherry array for visualization

## Hardware Integration Details

**Camera System:**
- Allied Vision Mako G-507 infrared cameras
- Resolution: 2463 x ~700 pixels (after cropping)
- Network configuration: PC 172.16.1.223, Camera 172.16.2.2
- Interface: Allied Vision Vimba SDK

**I/O Hardware:**
- USB data acquisition via uldaq library
- Encoder input for conveyor position tracking
- Digital outputs for light control and rejection latches
- Real-time synchronization with image capture

**Visualization System:**
- C++ Qt5/OpenGL application
- Subscribes to tracked cherry positions
- Projects color-coded circles: Red (Pit), Yellow (Maybe), Cyan (Side)
- Compensates for belt speed and system latency

## Configuration Parameters

**Detection Thresholds:**
```bash
ros2 param set /detection_server pick_threshold 0.5
ros2 param set /detection_server maybe_threshold <value>
```

**Algorithm Selection:**
```bash
ros2 param set /detection_server algorithm <name>
# Options: hdr_v1, hdr_v2, vote_v1, fasterRCNN-Mask_ResNet50_V1
```

**Image Saving:**
```bash
ros2 param set /image_services enable_saving True
ros2 param get /image_services base_path  # default: /home/user/Pictures/hdr/
```