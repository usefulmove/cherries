---
title: ChunkAccess VmbCPP CMake Build Configuration
description: CMake build configuration file for the ChunkAccess example demonstrating
  Vimba X C++ API chunk data access functionality.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/ChunkAccess/CMakeLists.txt
tags:
- cmake
- vimba-x
- chunk-access
- cpp-example
- build-configuration
related: []
last_analyzed: '2026-03-09T07:49:40Z'
---

# ChunkAccess VmbCPP CMake Build Configuration

This CMakeLists.txt file defines the build configuration for the ChunkAccess example application using the Vimba X C++ API (VmbCPP). It sets up a CMake project requiring version 3.0 or higher, finds the Vmb package with the CPP component, and creates an executable from main.cpp and AcquisitionHelperChunk source files. The configuration links against the Vmb::CPP library, requires C++11 standard compliance, and sets up Visual Studio debugger environment paths for proper DLL resolution during debugging.

**Key concepts:** `CMake project configuration`, `VmbCPP library integration`, `Chunk data access example`, `C++11 standard requirement`, `Visual Studio debugger configuration`
