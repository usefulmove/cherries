---
title: EventHandling VmbCPP CMakeLists
description: CMake build configuration for the VimbaX EventHandling example demonstrating
  camera event handling with the Vmb C++ API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/EventHandling/CMakeLists.txt
tags:
- cmake
- vimbax
- vmb-cpp
- event-handling
- camera-sdk
related: []
last_analyzed: '2026-03-09T07:49:45Z'
---

# EventHandling VmbCPP CMakeLists

This CMakeLists.txt file defines the build configuration for the EventHandling example project within the VimbaX SDK. It sets up a CMake project that builds an executable from main.cpp, EventHandling.cpp, EventHandling.h, EventObserver.cpp, and EventObserver.h source files. The configuration finds and links against the Vmb C++ library, requires C++11 standard compliance, and optionally includes hardcoded package location information when the example resides in its original install location. The Visual Studio debugger environment is also configured to include VMB binary directories in the PATH.

**Key concepts:** `CMake project configuration`, `Vmb C++ API integration`, `event handling example`, `camera SDK build setup`, `C++11 standard`
