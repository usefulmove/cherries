---
name: Vision Acquisition
layer: Vision
impact_area: Image Quality, Synchronization
---

# Vision Acquisition Layer

## Responsibility
Handles the capture of high-quality images from the Allied Vision (AVT) Vimba cameras. It manages exposure, gain, and the precise timing of image acquisition.

## Key Components

### 1. AVT Vimba Camera Node (`cherry_system/avt_vimba_camera/`)
*   **Type**: ROS2 Action Server / Lifecycle Node.
*   **Logic**:
    *   Receives a `Capture` action goal.
    *   Triggers the camera (software or hardware trigger).
    *   Retrieves the frame via the Vimba API.
    *   Publishes the raw image.
*   **Key Feature**: **HDR / Dual-Exposure**. The system may capture multiple exposures (short/long) to handle high-dynamic-range cherry surfaces (shiny reflections vs dark pits).

### 2. Camera Simulator (`cherry_system/camera_simulator/`)
*   **Purpose**: Allows development and testing without physical hardware.
*   **Logic**: Replays recorded bag files or generates synthetic frames when triggered.

## Design Decisions
*   **Action Interface**: Used instead of a simple topic to ensure the `control_node` knows exactly *when* an image was captured relative to the encoder tick.

## Discovery Links
*   **Code**: `src/cherry_system/avt_vimba_camera/`
*   **Reference**: [Vimba Interface Guide](./VIMBA_REFERENCE.md)
