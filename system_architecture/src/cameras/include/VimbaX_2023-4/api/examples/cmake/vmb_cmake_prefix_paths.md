---
title: VimbaX CMake Prefix Paths Configuration
description: CMake module that configures CMAKE_PREFIX_PATH with locations for finding
  the VimbaX SDK installation.
source: src/cameras/include/VimbaX_2023-4/api/examples/cmake/vmb_cmake_prefix_paths.cmake
tags:
- cmake
- vimbax
- sdk-configuration
- build-system
- camera-api
related: []
last_analyzed: '2026-03-09T07:50:31Z'
---

# VimbaX CMake Prefix Paths Configuration

This CMake module file assists in locating the VimbaX camera SDK by appending potential installation paths to CMAKE_PREFIX_PATH. It adds a hardcoded relative path (two directories up from the current file) and also checks for VMB_HOME and VIMBA_X_HOME environment variables to support flexible SDK installation locations. This enables CMake's find_package mechanism to locate VimbaX components.

**Key concepts:** `CMAKE_PREFIX_PATH configuration`, `VMB_HOME environment variable`, `VIMBA_X_HOME environment variable`, `SDK path detection`, `CMake list append operations`
