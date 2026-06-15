---
title: Cameras CMake Build Configuration
description: CMake build configuration for a ROS2 camera package that integrates with
  the Vimba SDK for Cognex cameras.
source: src/cameras/CMakeLists.txt
tags:
- cmake
- ros2
- vimba-sdk
- camera-drivers
- build-configuration
related: []
last_analyzed: '2026-03-09T07:46:34Z'
---

# Cameras CMake Build Configuration

This CMakeLists.txt file configures the build for a ROS2 'cameras' package that provides nodes for interfacing with Cognex cameras via the Vimba SDK (VimbaX_2023-4). It defines two executable targets: 'cognex_cam' for standard camera operation and 'cognex_hdr' for HDR camera functionality. The build integrates the Vimba SDK's C++ API (Vmb::CPP), locates required SDK components via CMake's find_package mechanism, and links against various ROS2 dependencies including rclcpp, sensor_msgs, and custom cherry_interfaces. The file includes significant commented-out code showing alternative approaches for finding and linking the Vimba libraries.

**Key concepts:** `Vimba SDK integration`, `ROS2 ament build system`, `Cognex camera nodes`, `VmbCPP library linking`, `HDR camera support`

## Sections

- **Project Setup** — Declares the 'cameras' project with CMake 3.8 minimum and configures compiler warning flags for GNU and Clang compilers.
- **Vimba SDK Configuration** — Locates and configures the Vimba SDK (VimbaX_2023-4) by finding the vmb-config.cmake file and using find_package to import the CPP component.
- **ROS2 Dependencies** — Finds required ROS2 packages including ament_cmake, rclcpp, cherry_interfaces, std_msgs, rclcpp_action, and sensor_msgs.
- **Executable Definitions** — Defines two executables: cognex_cam and cognex_hdr, both compiled from similar source files including main entry points, camera drivers, FrameObserver, and EventObserver.
- **Target Configuration** — Configures include directories for the targets and links them against ROS2 dependencies and the Vimba CPP library.
- **Installation** — Installs the cognex_cam and cognex_hdr executables to the package's lib directory.
- **Testing Configuration** — Configures ament_lint_auto for testing while skipping copyright and cpplint checks.
