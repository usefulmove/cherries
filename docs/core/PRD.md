# Cherry Processing System PRD

## Problem
Industrial cherry processing requires a high-speed, automated solution for detecting and rejecting cherries that contain pits. Manual inspection is inconsistent and slow. The system must process cherries in real-time on a moving conveyor belt and integrate with pneumatic rejection systems and robotic sorters.

## Goals
1. **Accuracy**: High recall for "Pit" class (target >93%) to ensure food safety.
2. **Efficiency**: High precision for "Clean" class to minimize the rejection of good fruit.
3. **Real-time Performance**: Maintain inference latency comparable to production baseline (~16ms on CPU) to ensure throughput.
4. **Iterative Improvement**: Establish a robust training system to refine models based on production data.

## Scope
**In scope:**
- 2-stage ML pipeline (Instance Segmentation + Binary Classification).
- ROS2-based control and coordination.
- Real-time visualization and tracking.
- Hybrid training workflow (Local development + Colab execution).

**Out of scope:**
- Redesign of the pneumatic hardware.
- Real-time training on the edge device.

## Constraints

### Hardware
- **Cameras:** Allied Vision infrared cameras (legacy: Mako G-319/G-507) → Cognex (current)
- **Sensors:** ULDAQ encoder for belt tracking
- **Actuation:** Pneumatic rejection valves, Fanuc robotic arm

### Compute Infrastructure

| Component | Hardware | Purpose |
|-----------|----------|---------|
| **Production System** | NVIDIA GPU (model TBD) + CPU | Real-time inference on edge device |
| **Training System** | Google Colab Pro (Tesla T4/A100) | Model training and experimentation |
| **Development System** | Local workstation (CPU) | Code development, relative benchmarking |

**Note:** Training requires GPU resources beyond local workstation capabilities, hence Colab Pro usage.

### Software
- **OS:** Ubuntu 22.04
- **Framework:** ROS2 Humble
- **ML:** PyTorch (with CUDA support for GPU inference)
- **Data:** Training datasets managed via Google Drive
