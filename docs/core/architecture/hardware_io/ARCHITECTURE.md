---
name: Hardware I/O
layer: Hardware
impact_area: Physical Interface, Timing
---

# Hardware I/O Layer

## Responsibility
Manages all low-level communication with physical hardware devices, excluding the cameras. This includes reading encoders, controlling pneumatic actuators, and managing strobe lights.

## Key Components

### 1. PLC Ethernet/IP Node (`threading_ws/src/plc_eip/`)
*   **Driver**: Interfaces with the **Omron/Keyence PLC** via EtherNet/IP (CIP).
*   **Responsibilities**:
    *   **Encoder Reading**: Reads high-speed counter (HSC) values for belt position.
    *   **I/O Control**: Manages digital inputs/outputs (latches, lights).
    *   **Synchronization**: Provides the time base for the entire system.

## Hardware Specs
*   **Encoder**: High-resolution rotary encoder attached to conveyor.
*   **Controller**: Industrial PLC (EtherNet/IP capable).
*   **Actuators**: Pneumatic valves controlled via PLC outputs.

## Discovery Links
*   **Code**: `threading_ws/src/plc_eip/`
