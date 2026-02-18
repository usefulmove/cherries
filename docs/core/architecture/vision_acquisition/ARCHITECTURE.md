---
name: Vision Acquisition
layer: Vision
impact_area: Image Quality, Synchronization
---

# Vision Acquisition Layer

## Responsibility
Handles the capture of high-quality images from the Cognex cameras (replaced AVT Vimba). It manages exposure, gain, and the precise timing of image acquisition via hardware triggers.

## Key Components

### 1. Cameras Node (`threading_ws/src/cameras/`)
*   **Type**: ROS2 Node (C++).
*   **Logic**:
    *   Receives triggers from `trigger_node`.
    *   Retrieves frames via the Cognex SDK.
    *   Publishes the raw images as `ImageLayer` messages.
*   **Key Feature**: **HDR / Multi-Layer**. Captures multiple exposures/layers (Top, Bot1, Bot2) to handle cherry surfaces.

### 2. Trigger Node (`threading_ws/src/trigger_node/`)
*   **Purpose**: Generates camera triggers based on encoder position.
*   **Logic**: Monitors encoder via `plc_eip` and fires triggers at fixed distance intervals.

## Design Decisions
*   **Hardware Triggering**: Critical for ensuring images align perfectly with encoder positions for tracking.

## Discovery Links
*   **Code**: `threading_ws/src/cameras/`
