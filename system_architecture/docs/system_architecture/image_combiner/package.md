---
title: ROS 2 Package Manifest – image_combiner
description: Documentation of the ROS 2 package.xml manifest for the image_combiner
  package, describing its metadata, runtime dependencies, test dependencies, and ament_python
  build system configuration.
source: docs/system_architecture/image_combiner/package.md
tags:
- ros2
- ament
- package-manifest
- robotics
- python
related: []
last_analyzed: '2026-03-09T07:44:26Z'
---

# ROS 2 Package Manifest – image_combiner

This document describes the ROS 2 package manifest (package.xml) for the 'image_combiner' package, which uses package format 3 and the ament_python build system. The package is versioned at 0.0.0, licensed under Apache-2.0, and maintained by Wesley. It declares runtime dependencies on 'cherry_interfaces' (a custom message/service interface package likely for a Cherry robot project) and 'sensor_msgs' (standard ROS sensor data messages). The manifest also specifies four test-time dependencies for enforcing code quality: ament_copyright, ament_flake8, ament_pep257, and python3-pytest.

**Key concepts:** `ROS 2 package format 3`, `ament_python build type`, `cherry_interfaces dependency`, `sensor_msgs dependency`, `test dependencies for code quality`, `Apache-2.0 license`

## Sections

- **ROS 2 Package Manifest – image_combiner** — Provides an overview of the image_combiner package manifest including its version, license, maintainer, build system, and dependency structure.
- **Dependencies** — Lists the runtime dependencies (cherry_interfaces, sensor_msgs) and test dependencies (ament_copyright, ament_flake8, ament_pep257, python3-pytest) for the package.
