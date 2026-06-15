---
title: Cognex CIC-5000 Camera ROS2 Node Header
description: Declares the Cic5000Camera ROS2 node class for controlling a Cognex CIC-5000
  machine-vision camera via the Vimba C++ SDK, providing image acquisition, action
  server integration, and PLC I/O coordination.
source: docs/system_architecture/cameras/include/cameras/cognex_cam.md
tags:
- ros2
- camera
- machine-vision
- vimba-sdk
- image-acquisition
related: []
last_analyzed: '2026-03-09T07:42:14Z'
---

# Cognex CIC-5000 Camera ROS2 Node Header

This document describes the Cic5000Camera class, a ROS2 node header that wraps an Allied Vision (Cognex CIC-5000 series) industrial camera using the VmbCPP (Vimba C++) SDK. The class manages the full lifecycle of dual-image acquisition including camera hardware initialization, event and frame observer registration, and orchestration of synchronized capture sequences producing separate top and bottom images. It uses condition variables to gate each pipeline stage, publishes images on dedicated ROS2 topics, exposes an rclcpp_action server for the cherry_interfaces Acquisition action, and handles PLC communication through ROS2 service clients and subscriptions for encoder-position-based triggering and light control.

**Key concepts:** `ROS2 node (rclcpp::Node) subclassing`, `rclcpp_action action server for triggered acquisition sequences`, `VmbCPP (Allied Vision Vimba C++ SDK) camera and frame management`, `Dual-frame capture (top and bottom views) with per-frame condition variables`, `PLC integration via cherry_interfaces services (ResetLatches, SetLights)`, `Encoder position subscriptions for positional triggering`, `Sensor image publishing (sensor_msgs/Image)`, `Camera event and frame observer callbacks`, `Gain configuration per image side (top/bottom)`

## Sections

- **Cognex CIC-5000 Camera ROS2 Node Header** — Provides an overview of the Cic5000Camera class, its role as a ROS2 node wrapping the VmbCPP SDK, dual-image acquisition workflow, condition variable synchronization, image publishing, action server, and PLC integration.
- **Exports** — Lists all public API elements including the Cic5000Camera class, constructor/destructor, camera initialization and control methods, frame callbacks for top/bottom capture, light and input callbacks, encoder subscription, and action server goal handling methods and type aliases.
- **Dependencies** — Enumerates the library dependencies: rclcpp, rclcpp_action, VmbCPP, cherry_interfaces, sensor_msgs, and std_msgs.
