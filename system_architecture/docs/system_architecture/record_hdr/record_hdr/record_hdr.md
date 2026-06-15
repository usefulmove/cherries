---
title: RecordHdr — ROS2 HDR Image Acquisition Action Client Node
description: A ROS2 action client node that listens to encoder position messages and
  triggers HDR image acquisition at fixed spatial intervals, saving the resulting
  multi-camera images and metadata to disk.
source: docs/system_architecture/record_hdr/record_hdr/record_hdr.md
tags:
- ros2
- action-client
- hdr-imaging
- computer-vision
- robotics
related: []
last_analyzed: '2026-03-09T07:45:20Z'
---

# RecordHdr — ROS2 HDR Image Acquisition Action Client Node

This document describes the RecordHdr ROS2 node, which functions as an action client for HDR image acquisition. The node subscribes to an encoder position topic using a SENSOR_DATA QoS profile and triggers acquisition goals whenever the encoder advances more than 190 mm. Upon goal completion, the node receives images from four cameras (top1, top2, bot1, bot2), converts them from ROS image messages to OpenCV format using CvBridge, saves them as BMP files, and writes JSON metadata containing encoder counts and millimeter values. The architecture uses an asynchronous callback chain for goal response and result handling.

**Key concepts:** `ROS2 action client (ActionClient)`, `Spatial-interval triggering via encoder position`, `HDR multi-camera image acquisition`, `CvBridge ROS image to OpenCV conversion`, `Asynchronous goal/result callback chain`, `JSON metadata serialization`, `QoS SENSOR_DATA profile subscription`

## Sections

- **RecordHdr — ROS2 HDR Image Acquisition Action Client Node** — Introduces the RecordHdr ROS2 node that acts as an action client for HDR image acquisition triggered by encoder position changes.
- **Exports** — Documents the RecordHdr class and its methods including initialization, goal sending, callbacks for goal response and results, image saving, and encoder handling.
- **Dependencies** — Lists the required packages: rclpy, cv_bridge, opencv-python (cv2), and cherry_interfaces.
