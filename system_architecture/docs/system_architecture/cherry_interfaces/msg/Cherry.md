---
title: Cherry ROS2 Message Definition
description: Documentation for a ROS2 custom message type that defines a Cherry message
  with 2D position coordinates and a type identifier byte for robotics applications.
source: docs/system_architecture/cherry_interfaces/msg/Cherry.md
tags:
- ros2
- message-definition
- robotics
- interfaces
- cherry-interfaces
related: []
last_analyzed: '2026-03-09T07:43:18Z'
---

# Cherry ROS2 Message Definition

This document describes the Cherry ROS2 message definition from the cherry_interfaces package. The message consists of three fields: two 32-bit floating-point values (x and y) representing 2D position coordinates, and a single byte (type) used to classify or differentiate cherry object variants. This message type is designed for use in ROS2 node graphs for publishing and subscribing to cherry-related data, commonly applied in agricultural robotics, object detection, or pick-and-place automation contexts.

**Key concepts:** `Cherry message type`, `2D position coordinates (x, y)`, `float32 data type`, `type identifier byte`, `ROS2 message interface`, `rosidl_default_generators`

## Sections

- **Cherry ROS2 Message Definition** — Introduces the Cherry.msg interface file from the cherry_interfaces package, explaining its three fields (x, y coordinates and type byte) and typical use cases in robotics applications.
- **Dependencies** — Lists the required ROS2 build system dependencies including rosidl_default_generators and built-in primitive types (float32, byte).
