---
title: ChunkAccess VmbC Example CMakeLists
description: CMake build configuration file for the ChunkAccess VmbC example, part
  of the VimbaX SDK camera API examples.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ChunkAccess/CMakeLists.txt
tags:
- cmake
- vimbax
- chunk-access
- camera-api
- build-configuration
related: []
last_analyzed: '2026-03-09T07:47:56Z'
---

# ChunkAccess VmbC Example CMakeLists

This CMakeLists.txt file defines the build configuration for the ChunkAccess example in the VimbaX 2023-4 SDK's VmbC API. It sets up a CMake project targeting C language, includes optional hardcoded package paths for the SDK location, finds the required Vmb package components, and creates an executable from main.c, ChunkAccessProg.c, and common source files. The target is linked against the Vmb::C library and VmbCExamplesCommon, compiled with C11 standard, and configured with Visual Studio debugger environment paths.

**Key concepts:** `CMake project configuration`, `VimbaX SDK integration`, `Vmb C API linking`, `Chunk data access example`, `C11 standard compilation`
