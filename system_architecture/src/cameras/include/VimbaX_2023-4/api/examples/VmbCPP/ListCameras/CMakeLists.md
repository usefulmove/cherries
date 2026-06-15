---
title: ListCameras VmbCPP CMake Build Configuration
description: CMake build configuration file for the ListCameras example application
  demonstrating VimbaX C++ API camera enumeration functionality.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/ListCameras/CMakeLists.txt
tags:
- cmake
- vimbax
- camera-api
- cpp-example
- build-configuration
related: []
last_analyzed: '2026-03-09T07:49:57Z'
---

# ListCameras VmbCPP CMake Build Configuration

This CMakeLists.txt file defines the build configuration for the ListCameras example application, which is part of the VimbaX 2023-4 SDK's C++ API examples. It sets up a CMake project targeting C++11, finds and links the Vmb CPP package, and compiles an executable from main.cpp, ListCameras.cpp, and ListCameras.h. The configuration includes optional loading of hardcoded package paths when the example resides in its original installation location, and sets up Visual Studio debugger environment paths for runtime library discovery.

**Key concepts:** `CMake project configuration`, `VimbaX SDK integration`, `C++ camera listing example`, `Vmb::CPP library linking`, `C++11 standard requirement`

## Sections

- **CMake Minimum Version and Project** — Specifies CMake 3.0 as minimum required version and defines the ListCameras project with C++ language support.
- **Vmb Package Discovery** — Conditionally includes cmake prefix paths and finds the required Vmb CPP package components for VimbaX SDK integration.
- **Executable Target Definition** — Creates the ListCameras_VmbCPP executable from source files and links it against the Vmb::CPP library.
- **Target Properties** — Configures C++11 standard compliance and Visual Studio debugger environment for proper DLL path resolution.
