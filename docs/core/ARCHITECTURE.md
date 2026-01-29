# Cherry Processing System Architecture

## Overview
The system uses a modular, service-oriented architecture built on **ROS2 Humble**. It employs a 2-stage deep learning pipeline for real-time cherry classification and sorting.

## System Data Flow
1. **Encoder Ticks**: `usb_io_node` reads the conveyor encoder and publishes changes.
2. **Trigger**: `control_node` monitors ticks and triggers image acquisition every ~500 ticks.
3. **Acquisition**: `avt_vimba_camera` (Action Server) strobes lights and captures top/bottom images.
4. **Combination**: `control_node` merges images into a 3-channel RGB composite (Top=Red, Bottom=Blue).
5. **Detection**: `cherry_detection` (Service Server) segments cherries and classifies them as Clean, Pit, Maybe, or Side.
6. **Tracking**: `control_node` updates the `frame_tracker` with detections, maintaining state as the belt moves.
7. **Visualization/Sorting**:
   - `tracking_projector` renders color-coded circles onto the belt.
   - `fanuc_comms` sends coordinates to a robot for sorting.
   - (Future) `usb_io_node` triggers pneumatic rejection.

## Core Components
| Component | Responsibility |
|-----------|----------------|
| `control_node` | Orchestration, frame tracking, and system logic. |
| `cherry_detection` | Inference server running Mask R-CNN and ResNet50. |
| `avt_vimba_camera`| Allied Vision camera driver and strobe control. |
| `usb_io` | Hardware I/O (Encoder, Lights, Pneumatics). |
| `tracking_projector` | OpenGL-based visual feedback. |

## Key Decisions
- **2-Stage Pipeline**: Decoupling segmentation from classification allows for simpler models and targeted training.
- **Service-Based Detection**: Decouples the compute-heavy AI from the time-sensitive control logic.
- **HDR Imaging**: Multiple exposure handling for consistent performance across lighting variations.

## Known Issues & Technical Debt
- **Model Loading Bug**: `cherry_detection` currently loads weights from `control_node/resource/` due to a config error.
- **Code Duplication**: `ai_detector.py` is duplicated across packages.
- **Dead Code**: `control_node` contains legacy detection files that are no longer executed.
- **Normalization Mismatch**: Missing ImageNet normalization in the inference pipeline.
