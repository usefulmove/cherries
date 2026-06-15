---
title: Fanuc Launch File
description: ROS 2 launch file documentation for the Fanuc communications package
  that starts all nodes required for a cherry-picking robotic system.
source: docs/system_architecture/fanuc_comms/launch/fanuc_launch.md
tags:
- ros2
- launch
- fanuc
- robotics
- orchestration
related: []
last_analyzed: '2026-03-09T07:44:10Z'
---

# Fanuc Launch File

This document describes the ROS 2 launch file (fanuc_launch.py) that serves as the top-level system launcher for a cherry-picking robotic application stack. The launch file instantiates eight active nodes across multiple packages including cherry_detection for object detection, cameras for Cognex HDR camera interface, plc_eip for PLC EtherNet/IP communication, cherry_buffer for position buffering, image_service for image services, system_monitor for temperature monitoring, trigger_node for hardware triggering, tracking_projector for laser/projector tracking, and fanuc_comms for Fanuc robot communication. The file resides in the fanuc_comms package but orchestrates the entire pipeline.

**Key concepts:** `ROS 2 LaunchDescription`, `Multi-node orchestration`, `Fanuc robot communication`, `Cherry detection pipeline`, `PLC EtherNet/IP integration`, `Tracking projector`, `System monitor`

## Sections

- **Fanuc Launch File** — Overview of the ROS 2 launch file that brings up the entire cherry-picking robotic application stack with eight active nodes spanning multiple packages.
- **Exports** — Documents the generate_launch_description() function that serves as the ROS 2 launch entry point returning a LaunchDescription with all required nodes.
- **Dependencies** — Lists the required dependencies: launch and launch_ros packages.
