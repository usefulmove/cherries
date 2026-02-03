---
name: Hardware I/O
layer: Hardware
impact_area: Physical Interface, Timing
---

# Hardware I/O Layer

## Responsibility
Manages all low-level communication with physical hardware devices, excluding the cameras. This includes reading encoders, controlling pneumatic actuators, and managing strobe lights.

## Key Components

### 1. USB I/O Node (`cherry_system/usb_io/`)
*   **Driver**: Interfaces with the **ULDAQ** (Universal Library for Data Acquisition) device.
*   **Responsibilities**:
    *   **Encoder Reading**: Publishes belt travel distance (ticks) to `control_node`.
    *   **Pneumatic Control**: Triggers air jets for rejection based on commands.
    *   **Strobe Triggering**: Hardware-timed output for camera synchronization.

## Hardware Specs
*   **Encoder**: High-resolution rotary encoder attached to the conveyor drive.
*   **IO Board**: Measurement Computing (MCC) USB-DAQ.
*   **Actuators**: High-speed pneumatic valves.

## Discovery Links
*   **Code**: `src/cherry_system/usb_io/`
