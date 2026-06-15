---
title: ROS Temperature Message Definition
description: A ROS (Robot Operating System) message definition file that declares
  a custom Temperature message type containing CPU and GPU temperature fields for
  hardware thermal monitoring.
source: docs/system_architecture/cherry_interfaces/msg/Temperature.md
tags:
- ros
- message-definition
- robotics
- telemetry
- temperature
related: []
last_analyzed: '2026-03-09T07:43:32Z'
---

# ROS Temperature Message Definition

This document describes a ROS message definition file belonging to the 'cherry_interfaces' package. It defines a custom 'Temperature' message type with two float32 fields: 'cpu' and 'gpu', intended to carry temperature readings (likely in degrees Celsius) from a computing platform's CPU and GPU. The message is compiled by ROS tooling (e.g., rosidl for ROS 2) into language-specific code (C++, Python) for use in nodes that publish or subscribe to hardware thermal data.

**Key concepts:** `ROS .msg interface definition`, `float32 cpu field for CPU temperature`, `float32 gpu field for GPU temperature`, `Custom message type for hardware thermal monitoring`, `rosidl message compilation`

## Sections

- **ROS Temperature Message Definition** — Describes the Temperature message type with cpu and gpu float32 fields for carrying hardware thermal readings in a ROS system.
- **Dependencies** — Lists the required dependencies including rosidl_default_generators and std_msgs for ROS message generation.
