---
title: VmbCPP Examples CMake Build Configuration
description: CMake configuration file for building VimbaX C++ API example projects,
  managing multiple sample applications with cross-platform installation support.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/CMakeLists.txt
tags:
- cmake
- vimbax
- camera-api
- build-configuration
- cpp-examples
related: []
last_analyzed: '2026-03-09T07:49:44Z'
---

# VmbCPP Examples CMake Build Configuration

This CMakeLists.txt file configures the build system for VimbaX C++ API example applications. It defines a function to register examples with optional aliases and ignore flags, then lists five example projects: AsynchronousGrab, SynchronousGrab, ChunkAccess, ListCameras, and EventHandling. The configuration supports selective building through VMB_CPP_EXAMPLES_LIST and individual ignore flags. It includes recursive functions to collect all build targets for installation, and handles platform-specific build file exclusions (Visual Studio project files on non-Windows, Xcode files on non-Apple platforms) during the installation process.

**Key concepts:** `CMake project configuration`, `VimbaX C++ API examples`, `Selective example building`, `Cross-platform build exclusions`, `Recursive target collection`, `Installation targets`

## Sections

- **Project Setup and Configuration Variables** — Defines the VmbCPPExamples project and sets up the VMB_CPP_EXAMPLES_LIST cache variable for selective example configuration.
- **Example Registration Function** — Defines vmb_cpp_example function that registers example directories with optional aliases and individual ignore flags.
- **Actual examples list** — Registers five VimbaX C++ examples: AsynchronousGrab, SynchronousGrab, ChunkAccess, ListCameras, and EventHandling.
- **overwrite list of examples set based on individual ignores** — Allows VMB_CPP_EXAMPLES_LIST to override individual example ignore settings with validation of example names.
- **finally add the necessary subdirectories** — Removes duplicate examples and adds each active example as a CMake subdirectory.
- **collect all targets for installation** — Defines recursive functions to collect all build targets and installs them to the bin directory.
- **exclude platform depending build files** — Handles platform-specific installation by excluding inappropriate IDE project files (Visual Studio on non-Windows, Xcode on non-Apple).
