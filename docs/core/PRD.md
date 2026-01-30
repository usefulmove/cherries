# Cherry Processing System PRD

## Problem
Industrial cherry processing requires a high-speed, automated solution for detecting and rejecting cherries that contain pits. Manual inspection is inconsistent and slow. The system must process cherries in real-time on a moving conveyor belt and integrate with pneumatic rejection systems and robotic sorters.

## Goals
1. **Accuracy**: High recall for "Pit" class (target >93%) to ensure food safety.
2. **Efficiency**: High precision for "Clean" class to minimize the rejection of good fruit.
3. **Real-time Performance**: Maintain inference latency under 30ms per frame (CPU) to match conveyor throughput.
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
- **Hardware**: Allied Vision infrared cameras, uldaq encoder, pneumatic actuators.
- **Compute**: Local edge system for inference; Google Colab Pro for training (due to local GPU limits).
- **Software**: ROS2 Humble, PyTorch, Ubuntu 22.04.
- **Data**: Training data managed via Google Drive.
