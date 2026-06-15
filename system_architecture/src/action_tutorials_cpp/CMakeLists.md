---
title: action_tutorials_cpp CMakeLists.txt
description: CMake build configuration file for the ROS 2 action tutorials C++ package,
  defining dependencies and build settings for action-based communication examples.
source: src/action_tutorials_cpp/CMakeLists.txt
tags:
- cmake
- ros2
- action-tutorials
- cpp
- build-configuration
related: []
last_analyzed: '2026-03-09T07:46:22Z'
---

# action_tutorials_cpp CMakeLists.txt

This CMakeLists.txt file configures the build system for the action_tutorials_cpp ROS 2 package. It requires CMake version 3.8 or higher and enables strict compiler warnings for GCC and Clang. The package depends on core ROS 2 libraries including ament_cmake, rclcpp, rclcpp_action for action server/client functionality, rclcpp_components for component-based nodes, and action_tutorials_interfaces for custom action message definitions. The testing section uses ament_lint_auto with copyright and cpplint checks disabled for development purposes.

**Key concepts:** `CMake build configuration`, `ROS 2 ament build system`, `action_tutorials_interfaces dependency`, `rclcpp_action for action communication`, `rclcpp_components for composable nodes`, `ament_lint_auto for automated testing`

## Sections

- **Project declaration** — Declares the CMake minimum version (3.8) and project name (action_tutorials_cpp).
- **Compiler options** — Enables strict compiler warnings (-Wall -Wextra -Wpedantic) for GCC and Clang compilers.
- **find dependencies** — Locates required ROS 2 packages including ament_cmake, action_tutorials_interfaces, rclcpp, rclcpp_action, and rclcpp_components.
- **BUILD_TESTING** — Configures automated testing with ament_lint_auto while skipping copyright and cpplint checks during development.
- **Package finalization** — Calls ament_package() to finalize the ROS 2 package configuration.
