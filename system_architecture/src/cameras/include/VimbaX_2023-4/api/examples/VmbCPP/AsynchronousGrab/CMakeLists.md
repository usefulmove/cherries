---
title: AsynchronousGrab CMake Configuration
description: CMake build configuration file for the AsynchronousGrab VmbCPP example,
  which demonstrates asynchronous image capture using the Vimba X C++ API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/AsynchronousGrab/CMakeLists.txt
tags:
- cmake
- vimbax
- camera-api
- build-configuration
- cpp-example
related: []
last_analyzed: '2026-03-09T07:49:31Z'
---

# AsynchronousGrab CMake Configuration

This CMakeLists.txt file configures the build for the AsynchronousGrab example application in the VimbaX 2023-4 SDK. It sets up a CMake project requiring version 3.0 or higher, finds the Vmb package with the CPP component, and creates an executable from main.cpp and AcquisitionHelper source files. The target is linked against Vmb::CPP and configured to use C++11 standard, with Visual Studio debugger environment path settings for proper DLL resolution.

**Key concepts:** `CMake project configuration`, `Vimba X C++ API integration`, `asynchronous image acquisition`, `target linking`, `C++11 standard requirement`

## Sections

- **CMake Configuration** — Defines the minimum CMake version (3.0) and project name (AsynchronousGrab) with C++ language support.
- **Package Discovery** — Conditionally includes VMB cmake prefix paths and finds the required Vmb CPP package components.
- **Target Definition** — Creates the AsynchronousGrab_VmbCPP executable from main.cpp and AcquisitionHelper source files, links it to Vmb::CPP, and sets C++11 standard with Visual Studio debugger environment configuration.
