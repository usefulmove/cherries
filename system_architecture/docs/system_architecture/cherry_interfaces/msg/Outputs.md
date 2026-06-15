---
title: 'ROS Message Definition: Outputs'
description: A ROS message file defining boolean output signals for the 'cherry' robot
  system, including named control outputs and generic digital output channels.
source: docs/system_architecture/cherry_interfaces/msg/Outputs.md
tags:
- ros
- robotics
- message-definition
- digital-outputs
- cherry
related: []
last_analyzed: '2026-03-09T07:43:31Z'
---

# ROS Message Definition: Outputs

This document describes the ROS message definition 'Outputs' from the 'cherry_interfaces' package, which represents 32 digital output signals for a robot platform. The message consists entirely of boolean fields, with the first three having semantic names (top_light for top lighting control, bot_light for bottom lighting control, and robot_latch for a mechanical or logical latch mechanism), while the remaining 29 fields (out3 through out31) are generic numbered digital output channels. This structure represents a hardware I/O board abstraction with 32 digital output lines, allowing ROS nodes to command actuators, lights, or other digital peripherals on the robot.

**Key concepts:** `ROS .msg format`, `Boolean fields`, `Named outputs (top_light, bot_light, robot_latch)`, `Generic digital outputs (out3 through out31)`, `Hardware I/O interface abstraction`, `Digital output channel mapping`

## Sections

- **ROS Message Definition: Outputs** — Documents the Outputs message type containing 32 boolean fields representing digital output signals, with three named outputs and 29 generic channels for robot hardware control.
