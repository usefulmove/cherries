---
title: ROS 2 Image Combiner Service Node
description: A ROS 2 service node that combines two grayscale camera images (top and
  bottom) into a single false-colour image by merging them as separate colour channels,
  with encoder-count-based vertical alignment via affine transformation.
source: docs/system_architecture/image_combiner/image_combiner/image_combiner.md
tags:
- ros2
- image-processing
- computer-vision
- opencv
- robotics
related: []
last_analyzed: '2026-03-09T07:44:25Z'
---

# ROS 2 Image Combiner Service Node

This documentation describes a ROS 2 service node called MinimalService (node name 'image_combiner') that exposes a ~/combine service of type CombineImages. When invoked, the service receives an ImageSet message containing two grayscale images (image_top and image_bot) along with encoder count values. It converts the ROS Image messages to OpenCV arrays using CvBridge, computes a vertical pixel offset using a configurable pixel_per_count scaling parameter (default ~0.2456), and applies cv2.warpAffine transformations to align the images vertically. The two aligned grayscale frames are then merged into a 3-channel false-colour image where the back image maps to the blue channel, zeros to green, and the top image to the red channel. The combined image is saved locally for debugging, published on the ~/combined_image topic, and returned in the service response. Error handling returns a black fallback image on failure.

**Key concepts:** `ROS 2 Node (rclpy)`, `Service server (CombineImages)`, `CvBridge image conversion`, `Affine/warp transformation for image alignment`, `Encoder-count-based pixel offset (scaling factor)`, `False-colour image merging (BGR channel fusion)`, `Image publisher for debugging`, `ROS 2 parameter declaration and retrieval`

## Sections

- **ROS 2 Image Combiner Service Node** — Overview of the image combiner node functionality including image conversion, alignment via affine transformation, and false-colour channel merging.
- **Exports** — Lists the exported class MinimalService, its methods (__init__, combine_callback, combine_images), and the main() entry point function.
- **Dependencies** — Lists required dependencies: rclpy, sensor_msgs, cherry_interfaces, cv_bridge, cv2 (opencv-python), and numpy.
