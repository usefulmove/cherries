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

*   **Custom Messages/Services**: (Agent should check `cherry_interfaces` directory for latest definitions)

## Key Constraints
*   **Real-time Constraint**: The entire pipeline (Trigger -> Sort) must happen within the travel time between camera and ejector.
*   **Inference Latency**: Must stay under **30ms** per frame to keep up with belt speed.
*   **ROS2 Distribution**: Humble Hawksbill.

## Discovery Links
*   [Global Standards](../../../core/STANDARDS.md)
*   [Legacy System Architecture](../../reference/legacy/cherry-processing-system-overview.md)
