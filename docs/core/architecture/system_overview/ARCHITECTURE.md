---
name: System Overview
layer: Global
impact_area: Architecture & Data Flow
---

# System Overview

## Responsibility
Defines the global architecture, message definitions (`cherry_interfaces`), and data flow constraints for the entire Cherry Processing System. It serves as the "contract" between different modules.

## System Data Flow (The "Life of a Cherry")

1.  **Transport**: Cherry moves on the conveyor belt.
2.  **Trigger**: `usb_io_node` (Hardware Layer) detects encoder movement.
3.  **Orchestration**: `control_node` (Tracking Layer) counts ticks and requests an image at a specific belt position.
4.  **Acquisition**: `avt_vimba_camera` (Vision Layer) fires strobe and captures Top/Bottom images.
5.  **Assembly**: Images are merged into a 3-channel composite (Red=Top, Blue=Bottom).
6.  **Inference**: `cherry_detection` (Inference Layer) runs segmentation (Mask R-CNN) and classification (ResNet50).
7.  **Tracking**: `control_node` maps the detection back to the moving belt coordinate system.
8.  **Action**:
    *   **Sorting**: Coordinates sent to Fanuc Robot via `fanuc_comms`.
    *   **Rejection**: Air jet triggered via `usb_io_node`.
    *   **Visualization**: State projected onto the belt via `tracking_projector`.

## Shared Interfaces (`cherry_interfaces`)
Located in `cherry_system/cherry_interfaces/`. This package contains the ROS2 definitions used by all nodes.

| File | Purpose | Used By |
|------|---------|----------|
| `msg/Cherry.msg` | Single cherry data (x, y, type) | All packages |
| `msg/CherryArray.msg` | Array of cherries | Fanuc, tracking |
| `msg/CherryArrayStamped.msg` | Timestamped cherry array | control_node, tracking |
| `msg/ImageSet.msg` | Paired top+bottom camera images with offsets | camera, control |
| `srv/Detection.srv` | Detect cherries in color image | cherry_detection |
| `srv/CombineImages.srv` | Merge top+bottom images | control_node |
| `srv/SetLights.srv` | Control top/bottom lights | usb_io |
| `srv/SetLatch.srv` | Set encoder latch state | usb_io |
| `srv/EncoderCount.srv` | Get current encoder count | usb_io |
| `action/Acquisition.action` | Asynchronous image acquisition | avt_camera, control |

## Key Constraints
*   **Real-time Constraint**: The entire pipeline (Trigger -> Sort) must happen within the travel time between camera and ejector.
*   **Inference Latency**: Must meet production throughput requirements (baseline: ~16ms on CPU, faster on GPU).
*   **ROS2 Distribution**: Humble Hawksbill.

## Launch Configurations

| Launch File | Purpose | Extra Nodes |
|-------------|---------|--------------|
| `conveyor_control_launch.py` | Full conveyor system | control_node |
| `fanuc_launch.py` | FANUC robot integration | fanuc_comms |

Both launch:
- `cherry_detection` (detection service)
- `avt_vimba_camera` (camera)
- `usb_io` (encoder + lights)
- 2x `tracking_projector` (dual projectors) with position parameters

## Cherry Classification Types

| Code | Label | Description | Action |
|------|-------|-------------|---------|
| 0 | None | Background | Ignore |
| 1 | Clean cherry | Good cherry | Pass through (not sent to robot) |
| 2 | Pitted cherry | Defective | Send to robot for sorting |
| 3 | Side | Cherry on edge of conveyor | Track but don't sort |
| 5 | Maybe | Uncertain classification | Yellow visual marker |

## Discovery Links
*   [Global Standards](../../../core/STANDARDS.md)
*   [Known Issues](../../../reference/known-issues.md)
