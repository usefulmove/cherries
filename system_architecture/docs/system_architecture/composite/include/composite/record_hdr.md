---
title: RecordHdr — ROS 2 HDR Image Acquisition & Cherry Detection Node Header
description: Declares the RecordHdr ROS 2 node class, which orchestrates HDR image
  acquisition, image combination, and cherry detection by acting as an action server
  for FindCherries requests and coordinating multiple action/service clients.
source: docs/system_architecture/composite/include/composite/record_hdr.md
tags:
- ros2
- computer-vision
- action-server
- image-processing
- cherry-detection
related: []
last_analyzed: '2026-03-09T07:44:01Z'
---

# RecordHdr — ROS 2 HDR Image Acquisition & Cherry Detection Node Header

This document describes the RecordHdr class header file, which defines a ROS 2 node that serves as the central coordinator for an HDR cherry-detection pipeline. The class exposes a FindCherries action server allowing upstream orchestrators to trigger detection runs, while internally it manages an HDR Acquisition action client to collect image frames from a camera, invokes CombineImages and Detection services to merge and analyze the frames, and uses a Trigger service client to synchronize with a PLC during image capture. The node follows standard ROS 2 action server patterns with lifecycle callbacks for goal handling, cancellation, and execution, and integrates OpenCV via cv_bridge and image_transport for image manipulation.

**Key concepts:** `rclcpp::Node subclass`, `ROS 2 action server (FindCherries)`, `ROS 2 action client (Acquisition/HDR)`, `ROS 2 service clients (Trigger, CombineImages, Detection)`, `HDR image acquisition pipeline`, `Goal/cancel/accept lifecycle callbacks`, `cv_bridge and image_transport integration`, `cherry_interfaces custom messages/actions/services`

## Sections

- **RecordHdr — ROS 2 HDR Image Acquisition & Cherry Detection Node Header** — Introduces the RecordHdr class as a ROS 2 node that orchestrates HDR image acquisition and cherry detection through action servers, action clients, and service clients.
- **Exports** — Documents the class declaration, constructor, action server callbacks (handle_goal, handle_cancel, handle_accepted, execute), action client callbacks, and methods for image acquisition, saving, PLC reset, image combination, and detection.
- **Dependencies** — Lists required dependencies including rclcpp, rclcpp_action, image_transport, cv_bridge, OpenCV, sensor_msgs, and cherry_interfaces.
