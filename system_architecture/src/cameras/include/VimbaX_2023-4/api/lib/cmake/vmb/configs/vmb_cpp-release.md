---
title: VimbaX C++ CMake Release Configuration
description: CMake target import file that configures the Vmb::CPP library target
  for Release builds, specifying the shared library location.
source: src/cameras/include/VimbaX_2023-4/api/lib/cmake/vmb/configs/vmb_cpp-release.cmake
tags:
- cmake
- vimba
- camera-sdk
- release-config
- build-system
related: []
last_analyzed: '2026-03-09T07:53:00Z'
---

# VimbaX C++ CMake Release Configuration

This is a CMake configuration file generated for the VimbaX SDK (version 2023-4) that defines the Release build configuration for the Vmb::CPP target. It specifies the imported library location as libVmbCPP.so and sets up the necessary CMake properties for linking against the Vimba C++ API shared library. This file is typically auto-generated and consumed by CMake's find_package() mechanism to provide proper target linking information for camera applications using the Allied Vision VimbaX SDK.

**Key concepts:** `CMake imported targets`, `Vmb::CPP library target`, `Release configuration`, `Shared library linking (libVmbCPP.so)`, `CMake IMPORT_PREFIX variable`

## Dependencies

`libVmbCPP.so`
