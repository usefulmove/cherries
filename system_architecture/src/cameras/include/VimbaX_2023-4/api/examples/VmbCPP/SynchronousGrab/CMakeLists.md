---
title: SynchronousGrab CMake Configuration
description: CMake build configuration for the VmbCPP SynchronousGrab example application,
  which demonstrates synchronous image acquisition using the Vimba X C++ API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/SynchronousGrab/CMakeLists.txt
tags:
- cmake
- vimba-x
- camera-api
- build-configuration
- cpp-example
related: []
last_analyzed: '2026-03-09T07:50:05Z'
---

# SynchronousGrab CMake Configuration

This CMakeLists.txt file defines the build configuration for the SynchronousGrab example application, which is part of the VmbCPP (Vimba X C++) API examples. The configuration requires CMake 3.0 or higher, sets up the project with C++ language support, and conditionally includes VMB cmake prefix paths if the example is in its original install location. It locates the Vmb package with CPP components, creates an executable target from main.cpp, SynchronousGrab.cpp, and SynchronousGrab.h source files, links against the Vmb::CPP library, and configures the target to use C++11 standard with Visual Studio debugger environment paths set appropriately.

**Key concepts:** `CMake project configuration`, `Vimba X C++ API integration`, `synchronous image grabbing`, `target linking with Vmb::CPP`, `C++11 standard requirement`

## Sections

- **CMake Version and Project** — Specifies CMake minimum version 3.0 and declares the SynchronousGrab project with C++ language support.
- **VMB Prefix Paths** — Conditionally includes VMB cmake prefix paths if the example is in its original install location.
- **Package Discovery** — Finds the required Vmb package with CPP components using multiple package name variants.
- **Target Definition** — Creates the SynchronousGrab_VmbCPP executable target from main.cpp, SynchronousGrab.cpp, and SynchronousGrab.h.
- **Linking and Properties** — Links the target against Vmb::CPP library and sets C++11 standard and Visual Studio debugger environment.
