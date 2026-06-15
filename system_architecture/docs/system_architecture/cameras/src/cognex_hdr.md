---
title: Cognex CIC5000 HDR Camera ROS2 Node
description: Implements a ROS2 node (Cic5000CameraHdr) that drives an Allied Vision
  GigE camera via the VmbCPP SDK to capture HDR image sequences with configurable
  gain/exposure per frame, publishing results as image sets and supporting both action-server
  and topic-based triggering.
source: docs/system_architecture/cameras/src/cognex_hdr.md
tags:
- ros2
- camera-driver
- hdr-imaging
- vimba-sdk
- gige-vision
related: []
last_analyzed: '2026-03-09T07:42:25Z'
---

# Cognex CIC5000 HDR Camera ROS2 Node

This documentation describes the Cic5000CameraHdr ROS2 node implementation for interfacing with a Cognex/Allied Vision CIC-5000 GigE camera using the VmbCPP (Vimba C++) SDK. The node supports HDR image capture sequences with up to six exposure layers (top1/2/3 and bot1/2/3), each with configurable gain and exposure settings. Capture sequences can be triggered via topic-based triggers (cherry_interfaces/Trigger) or a ROS2 action server (Acquisitionhdr). The implementation includes GigE packet size negotiation, frame buffer allocation, per-frame observer callbacks, pixel format conversion to ROS image encodings, and PLC integration for controlling top and back lights during capture. Captured frames include embedded encoder count and position metadata and are published both individually and as aggregated ImageSetHdr messages.

**Key concepts:** `Allied Vision VmbCPP SDK integration`, `HDR multi-exposure capture sequence`, `ROS2 action server (Acquisitionhdr action)`, `GigE packet size auto-adjustment`, `Software trigger synchronization via condition variables`, `Per-frame gain and exposure control`, `VmbPixelFormat to sensor_msgs encoding conversion`, `FrameObserver and EventObserver callback pattern`, `PLC integration via ResetLatches and SetLights services`, `ImageSetHdr multi-layer image publishing`, `Encoder count subscription for positional metadata`, `ExposureEnd event notification via feature observer`

## Sections

- **Cognex CIC5000 HDR Camera ROS2 Node** — Provides an overview of the Cic5000CameraHdr ROS2 node that interfaces with an Allied Vision CIC-5000 GigE camera for HDR image capture sequences using the VmbCPP SDK.
- **Exports** — Documents the exported functions, classes, and methods including GigE packet adjustment, frame-to-image conversion, the Cic5000CameraHdr node class, initialization/destruction methods, capture sequence logic, trigger/encoder callbacks, action server handlers, and per-layer frame callbacks.
- **Dependencies** — Lists the external dependencies required by the node: VmbCPP (Allied Vision Vimba C++ SDK), rclcpp, rclcpp_action, sensor_msgs, std_msgs, and cherry_interfaces.
