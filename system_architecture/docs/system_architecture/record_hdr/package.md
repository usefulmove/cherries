---
title: ROS 2 Package Manifest – record_hdr
description: Documentation of the ROS 2 package.xml manifest (format 3) for the record_hdr
  Python package, declaring its metadata, runtime dependencies, test dependencies,
  and build system.
source: docs/system_architecture/record_hdr/package.md
tags:
- ros2
- ament
- package-manifest
- python
- robotics
related: []
last_analyzed: '2026-03-09T07:45:13Z'
---

# ROS 2 Package Manifest – record_hdr

This document describes the ROS 2 package manifest for the record_hdr package, a pure-Python ROS 2 package built with the ament_python build system. It uses package format 3 and declares two runtime dependencies: rclpy (the standard ROS 2 Python client library) and cherry_interfaces (a custom interfaces package for project-specific ROS messages or services). The package is licensed under Apache-2.0 and includes standard ROS 2 Python test tooling such as ament_copyright, ament_flake8, ament_pep257, and python3-pytest, indicating adherence to ROS 2 coding style conventions. The version 0.0.0 and placeholder description suggest the package is in early development.

**Key concepts:** `ROS 2 package manifest`, `ament_python build system`, `package format 3`, `rclpy runtime dependency`, `cherry_interfaces custom interfaces`, `Apache-2.0 license`, `ROS 2 test tooling`

## Sections

- **ROS 2 Package Manifest – record_hdr** — Overview of the record_hdr package manifest including its metadata, build system, dependencies, licensing, and early development status.
- **Dependencies** — Lists the runtime dependencies (rclpy, cherry_interfaces) and test dependencies (ament_copyright, ament_flake8, ament_pep257, python3-pytest) for the package.
