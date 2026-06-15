---
title: ROS2 Action Tutorials C++ Package Manifest
description: Package.xml manifest file for a ROS2 C++ package demonstrating action
  server/client tutorials using rclcpp_action.
source: src/action_tutorials_cpp/package.xml
tags:
- ros2
- package-manifest
- cpp
- action-server
- ament-cmake
related: []
last_analyzed: '2026-03-09T07:46:23Z'
---

# ROS2 Action Tutorials C++ Package Manifest

This is a ROS2 package manifest file (package.xml) for the action_tutorials_cpp package, which is a C++ tutorial package for learning ROS2 action servers and clients. The package uses the ament_cmake build system and depends on core ROS2 libraries including rclcpp (ROS C++ client library), rclcpp_action (for action server/client functionality), rclcpp_components (for composable node support), and action_tutorials_interfaces (which likely contains the custom action message definitions). The package is configured with format 3 schema and includes standard linting tools for testing.

**Key concepts:** `ROS2 package format 3`, `ament_cmake build system`, `rclcpp_action for action communication`, `rclcpp_components for composable nodes`, `action_tutorials_interfaces dependency`

## Dependencies

`ament_cmake`, `action_tutorials_interfaces`, `rclcpp`, `rclcpp_action`, `rclcpp_components`, `ament_lint_auto`, `ament_lint_common`

## Sections

- **Package Metadata** — Package name, version, description, maintainer, and license information
- **Build Tool Dependencies** — Specifies ament_cmake as the build tool
- **Runtime Dependencies** — Core ROS2 C++ libraries and action interfaces required at runtime
- **Test Dependencies** — Linting tools for automated testing
- **Export Configuration** — Exports build type as ament_cmake
