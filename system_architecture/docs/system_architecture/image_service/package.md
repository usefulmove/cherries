---
title: ROS 2 Package Manifest – image_service
description: ROS 2 package.xml manifest (format 3) for the image_service package,
  declaring its metadata, runtime dependency on cherry_interfaces, and build/test
  tooling configuration.
source: docs/system_architecture/image_service/package.md
tags:
- ros2
- ament
- package-manifest
- robotics
- python
related: []
last_analyzed: '2026-03-09T07:44:44Z'
---

# ROS 2 Package Manifest – image_service

This document describes the ROS 2 package manifest (package.xml, format 3) for the image_service package. The package is maintained by 'wesley' under the Apache-2.0 license and is implemented in Python using the ament_python build system. It declares a runtime dependency on the cherry_interfaces package (a custom ROS 2 interfaces package for the same project) and specifies standard ROS 2 Python linting and testing tools (ament_copyright, ament_flake8, ament_pep257, python3-pytest) as test dependencies.

**Key concepts:** `ROS 2 package format 3`, `ament_python build type`, `Runtime dependency on cherry_interfaces`, `Apache-2.0 license`, `Test dependencies: ament_copyright, ament_flake8, ament_pep257, python3-pytest`

## Sections

- **ROS 2 Package Manifest – image_service** — Provides an overview of the image_service package manifest including its maintainer, license, build system (ament_python), and runtime/test dependencies.
- **Dependencies** — Lists the package dependencies including cherry_interfaces for runtime and ament_copyright, ament_flake8, ament_pep257, and python3-pytest for testing.
