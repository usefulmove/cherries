---
title: ROS 2 PickMode Message Definition
description: Documentation for a ROS 2 custom message definition that declares a single
  integer field 'mode' for representing a pick operation mode in robotic systems.
source: docs/system_architecture/cherry_interfaces/msg/PickMode.md
tags:
- ros2
- message
- robotics
- interface
- cherry-interfaces
related: []
last_analyzed: '2026-03-09T07:43:33Z'
---

# ROS 2 PickMode Message Definition

This document describes the PickMode message definition from the cherry_interfaces ROS 2 package. The message contains a single field 'mode' of type int64, which is used to communicate or command different pick operation modes in a robotic system. The message enables coordination between ROS 2 nodes for pick actions, potentially distinguishing between different picking strategies, states, or behaviors such as manual vs. automatic modes or different grip styles. The message can be published/subscribed to over ROS 2 topics or used in service/action definitions.

**Key concepts:** `ROS 2 .msg file format`, `int64 data type`, `Single-field message`, `Pick mode selection`, `Custom interface definition`, `rosidl message generation`

## Sections

- **ROS 2 PickMode Message Definition** — Describes the purpose and structure of the PickMode message, which contains a single int64 'mode' field for coordinating pick operations across ROS 2 nodes.
- **Dependencies** — Lists the required dependencies including the ROS 2 message build system (rosidl/ament_cmake) and the parent cherry_interfaces package.
