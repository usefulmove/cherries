---
title: ROS 2 Package Manifest – tracking_projector
description: Documentation of the ROS 2 package manifest (package.xml) for the tracking_projector
  package, which tracks items on a conveyor belt and displays information via a projector.
source: docs/system_architecture/tracking_projector/package.md
tags:
- ros2
- ament-cmake
- package-xml
- robotics
- qt6
related: []
last_analyzed: '2026-03-09T07:45:56Z'
---

# ROS 2 Package Manifest – tracking_projector

This document describes the ROS 2 package manifest (package.xml, format 3) for the 'tracking_projector' package version 1.4.2. The package implements a ROS 2 node designed to track items moving on a conveyor belt and display information via a projector. It uses the ament_cmake build toolchain and declares dependencies on core ROS 2 libraries (rclcpp, std_msgs) for node communication and messaging, as well as Qt 6 libraries (qtbase6-dev, qt6-qmake for build; libqt6-core for runtime) for graphical/projection functionality. The package is maintained by Wesley Havener at Saber Engineering under the Apache License 2.0.

**Key concepts:** `tracking_projector package`, `ROS 2 package format 3`, `ament_cmake build system`, `conveyor belt item tracking`, `projector output`, `rclcpp and std_msgs dependencies`, `Qt 6 graphical libraries`, `Apache License 2.0`

## Sections

- **ROS 2 Package Manifest – tracking_projector** — Provides an overview of the tracking_projector package manifest including its purpose, version, build system, maintainer, and license information.
- **Dependencies** — Lists all build and runtime dependencies including ament_cmake, rclcpp, std_msgs, and Qt6 libraries.
