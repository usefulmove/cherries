---
title: ImageSave ROS Service Definition
description: A ROS 2 service definition file that specifies a service for saving images,
  accepting a sensor image as input and returning a result code indicating the outcome.
source: docs/system_architecture/cherry_interfaces/srv/ImageSave.md
tags:
- ros2
- service
- image-processing
- robotics
- interface
related: []
last_analyzed: '2026-03-09T07:43:46Z'
---

# ImageSave ROS Service Definition

This document describes the ImageSave ROS 2 service definition from the cherry_interfaces package. The service is designed for saving images in the context of a cherry-detection robotic system. It defines a request containing a sensor_msgs/Image field for transmitting raw image data to the service server, and a response containing an int64 result_code field that returns a numeric status code indicating success, failure, or possibly the number of detected cherries. The document follows the standard ROS .srv file convention with the three-dash separator dividing request and response sections.

**Key concepts:** `ROS 2 .srv service definition`, `sensor_msgs/Image request field`, `int64 result_code response field`, `three-dash separator for request/response`, `cherry_interfaces package`

## Sections

- **ImageSave ROS Service Definition** — Explains the purpose and structure of the ImageSave service definition, including its request field (sensor_msgs/Image) and response field (int64 result_code) separated by the standard ROS three-dash convention.
- **Dependencies** — Lists the service's dependency on the sensor_msgs/Image message type for the request field.
