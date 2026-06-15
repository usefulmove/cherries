---
title: ROS 2 Trigger Message Definition
description: A ROS 2 custom message definition for a trigger event, carrying a frame
  identifier, timestamp, and encoder positional data for synchronization or event-driven
  robotics workflows.
source: docs/system_architecture/cherry_interfaces/msg/Trigger.md
tags:
- ros2
- robotics
- message-definition
- encoder
- trigger
related: []
last_analyzed: '2026-03-09T07:43:34Z'
---

# ROS 2 Trigger Message Definition

This document describes a ROS 2 custom message definition (.msg) from the 'cherry_interfaces' package that defines a 'Trigger' message. The message contains four fields: a 64-bit integer 'frame_id' for identifying or sequencing trigger events, a 'builtin_interfaces/Time' stamp for recording creation time, an 'encoder_count' for raw encoder pulses, and an 'encoder_mm' for encoder-derived position in millimeters. This message type is designed to synchronize sensor data capture (such as camera frame triggers) with mechanical position feedback from encoders, supporting industrial inspection, conveyor-belt imaging, or precision motion systems in ROS 2.

**Key concepts:** `frame_id: int64 identifier for trigger frame or event sequence`, `stamp: ROS 2 built-in time type (builtin_interfaces/Time)`, `encoder_count: int64 raw encoder tick/pulse count`, `encoder_mm: int64 encoder position in millimeters`, `Custom ROS 2 .msg interface file for generating language-specific message classes`

## Sections

- **ROS 2 Trigger Message Definition** — Introduces the Trigger message and explains its purpose for synchronizing trigger events with encoder position data in robotics applications.
- **Dependencies** — Lists the builtin_interfaces/Time dependency required by the Trigger message for timestamp functionality.
