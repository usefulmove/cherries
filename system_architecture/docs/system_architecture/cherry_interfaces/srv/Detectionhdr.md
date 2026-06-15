---
title: ROS 2 HDR Cherry Detection Service Definition
description: A ROS 2 service interface definition (.srv) for an HDR cherry detection
  service, accepting six HDR-bracketed stereo camera images with encoder metadata
  and returning detected cherry data with keypoint overlays.
source: docs/system_architecture/cherry_interfaces/srv/Detectionhdr.md
tags:
- ros2
- service-interface
- hdr-imaging
- cherry-detection
- computer-vision
related: []
last_analyzed: '2026-03-09T07:43:37Z'
---

# ROS 2 HDR Cherry Detection Service Definition

This document describes the Detectionhdr.srv ROS 2 service definition used in the cherry_interfaces package for HDR-based cherry detection. The service request accepts a frame identifier, six images representing HDR bracket sequences from dual stereo cameras (three top-camera exposures and three bottom-camera exposures), along with encoder counts and millimeter measurements for positional metadata. The response returns a keypoint overlay image, an array of detected Cherry objects, and an encoder count for downstream synchronization. This service is designed for precision agricultural robotics using HDR multi-exposure imaging to robustly detect cherries under varying lighting conditions.

**Key concepts:** `ROS 2 .srv file format with request/response separated by '---'`, `HDR imaging with three bracketed exposures per camera`, `Dual-camera stereo setup (top and bottom cameras)`, `sensor_msgs/Image for raw image transport`, `Encoder counts (int64) for positional metadata at acquisition time`, `Millimeter measurements corresponding to encoder positions`, `frame_id (uint64) for identifying the processing frame`, `cherry_interfaces/Cherry[] array for detected cherries`, `Keypoint image with visual overlay of detections`

## Sections

- **ROS 2 HDR Cherry Detection Service Definition** — Describes the complete service definition including request fields for HDR images, encoder metadata, and response fields for detection results.
- **Dependencies** — Lists the required message types: sensor_msgs/Image and cherry_interfaces/Cherry.
