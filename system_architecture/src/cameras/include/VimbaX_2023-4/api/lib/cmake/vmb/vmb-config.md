---
title: VimbaX SDK CMake Package Configuration
description: CMake package configuration file for the Allied Vision VimbaX 2023-4
  SDK, enabling find_package() integration for Vimba camera SDK components.
source: src/cameras/include/VimbaX_2023-4/api/lib/cmake/vmb/vmb-config.cmake
tags:
- cmake
- vimba
- sdk-configuration
- camera-api
- build-system
related: []
last_analyzed: '2026-03-09T07:53:25Z'
---

# VimbaX SDK CMake Package Configuration

This CMake configuration file provides the find_package() integration for the Allied Vision VimbaX 2023-4 camera SDK. It implements component-based discovery for VmbC (C API), VmbCPP (C++ API), VmbCPP sources, and VmbImageTransform libraries. The file handles dependency resolution between components (e.g., VmbCPP depends on VmbC), validates platform compatibility (Windows/Unix), and sets up proper include paths for the SDK. It dynamically includes the appropriate component-specific CMake configuration files based on the requested components.

**Key concepts:** `CMake find_package configuration`, `VimbaX SDK components (VmbC, VmbCPP, VmbImageTransform)`, `Component dependency management`, `Cross-platform support (Windows/Unix)`, `Package discovery and inclusion`

## Dependencies

`CMake >= 3.0`, `vmb_c.cmake`, `vmb_cpp.cmake`, `vmb_cpp_sources.cmake`, `vmb_imagetransform.cmake`

## Sections

- **Package Initialization** — Standard CMake package config macros for path validation and component checking
- **Find Package Implementation** — Main implementation function that discovers and loads VimbaX SDK components
- **Component Dependencies** — Defines dependencies between SDK components (cpp depends on c, etc.)
- **Component Loading** — Includes all discovered component configuration files
