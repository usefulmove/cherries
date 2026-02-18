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
2.  **Trigger**: `plc_eip` (Hardware Layer) tracks encoder; `trigger_node` fires cameras.
3.  **Acquisition**: `cameras` (Vision Layer) captures Top/Bot1/Bot2 images (HDR).
4.  **Assembly**: `image_combiner` aligns images based on encoder values into `ImageSetHdr`.
5.  **Orchestration**: `composite` (Tracking Layer) manages the pipeline via Actions.
6.  **Inference**: `cherry_detection` (Inference Layer) runs 3-model pipeline (Seg, Class, Stem).
7.  **Tracking**: `composite` maps detections to belt coordinates.
8.  **Action**:
    *   **Sorting**: Coordinates sent to Fanuc Robot via `fanuc_comms`.
    *   **Rejection**: Air jet triggered (via PLC/IO).
    *   **Visualization**: State projected onto the belt via `tracking_projector`.

## Shared Interfaces (`cherry_interfaces`)
Located in `threading_ws/src/cherry_interfaces/`. This package contains the ROS2 definitions used by all nodes.

| File | Purpose | Used By |
|------|---------|----------|
| `msg/Cherry.msg` | Single cherry data (x, y, type) | All packages |
| `msg/ImageSetHdr.msg` | Multi-layer HDR image set | cameras, detection |
| `srv/Detectionhdr.srv` | Detect cherries in HDR set | cherry_detection |
| `action/FindCherries.action` | Main orchestration action | composite |

## Key Constraints
*   **Real-time Constraint**: The entire pipeline must happen within the travel time between camera and ejector.
*   **Inference Latency**: Must meet production throughput requirements (GPU enabled).
*   **ROS2 Distribution**: Humble Hawksbill.

## Launch Configurations

| Launch File | Purpose | Extra Nodes |
|-------------|---------|--------------|
| `production_launch.py` | Full production system | composite, cameras, detection |

(Note: Exact launch file names in `threading_ws` may vary, check package `launch/` directories)

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
