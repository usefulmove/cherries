# Cherry Processing Machine - System Overview

## What This System Does

This is an **automated cherry pit detection and rejection system** that uses machine learning and computer vision to classify cherries on a moving conveyor belt, then physically removes cherries that still contain pits through pneumatic actuation.

**System Actions by Classification:**
- **Clean**: Allows cherry to continue on belt (green projection)
- **Pit**: Physically ejects cherry via pneumatic actuator (red projection)
- **Maybe**: Allows cherry to continue for manual inspection (yellow projection)  
- **Side**: Filters out due to unreliable positioning (cyan projection)

## Architecture at a Glance

Built on **ROS2 Humble** with a modular, service-oriented architecture:

**Key Packages:**
- **`control_node`**: System orchestrator - manages timing, tracking, and coordination
- **`cherry_detection`**: AI engine - runs 2-stage ML pipeline for detection/classification
- **`usb_io`**: Hardware interface - reads encoder, controls lights/rejection latches
- **`tracking_projector`**: Visual feedback - projects color-coded results onto cherries
- **`cherry_interfaces`**: Custom message/service definitions for inter-node communication

**Hardware Components:**
- Allied Vision infrared cameras (top + bottom views)
- Encoder for conveyor position tracking
- Pneumatic rejection system for pit-containing cherries
- Projector for real-time visual feedback

## How It Works (The Cherry Journey)

1. **Movement Detection**: Encoder tracks conveyor belt position, publishing distance changes
2. **Image Capture**: Every ~500 encoder ticks (~belt segment), system triggers HDR image acquisition
3. **AI Processing**: Combined top/bottom images processed through 2-stage ML pipeline:
   - Stage 1: Mask R-CNN segments individual cherries
   - Stage 2: ResNet50 classifies each cherry (Clean/Pit/Maybe/Side)
4. **Coordinate Transformation**: Pixel coordinates converted to real-world meters relative to machine origin
5. **Tracking & Visualization**: Cherry positions tracked as belt moves, with color-coded projection
6. **Rejection**: When tracked cherry reaches rejection point, pneumatic actuator ejects pit-containing cherries

## Key Technical Concepts

**Coordinate Transformation**: Critical for accurate tracking - converts pixel space to real-world meters using predefined scaling, rotation, and origin offset parameters.

**Frame Tracking**: Maintains cherry state as they move down the belt, ensuring system knows exact position even after cherries leave camera view.

**Service Architecture**: Modular design using ROS2 services and actions for clean separation between image acquisition, detection, and actuation.

**HDR Imaging**: Multiple exposure levels handle varying cherry appearances and lighting conditions for consistent detection performance.
