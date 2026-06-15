---
title: ActionCommands VmbC Example CMakeLists
description: CMake build configuration file for the ActionCommands example application
  demonstrating VimbaX SDK action commands functionality in C.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ActionCommands/CMakeLists.txt
tags:
- cmake
- vimbax
- vmbc
- action-commands
- camera-api
related: []
last_analyzed: '2026-03-09T07:46:29Z'
---

# ActionCommands VmbC Example CMakeLists

This CMakeLists.txt file defines the build configuration for the ActionCommands example application, which is part of the VimbaX 2023-4 SDK's C API examples. The project is configured as a C11 application that links against the VmbC library and common example utilities. It includes source files for action command handling, image acquisition, and helper functions. The build system supports both Windows and Unix platforms, with pthread linking on Unix systems and Visual Studio debugger environment configuration on Windows.

**Key concepts:** `CMake build configuration`, `VimbaX SDK integration`, `Action Commands example`, `C language project`, `VmbC library linking`, `Cross-platform build support`

## Sections

- **Project Configuration** — Defines the CMake minimum version (3.0) and project name (ActionCommands) with C language specification.
- **Vmb Package Discovery** — Includes optional cmake prefix paths and finds the required Vmb package with C component support.
- **Common Library Setup** — Conditionally adds the VmbCExamplesCommon subdirectory if not already defined as a target.
- **Executable Definition** — Creates the ActionCommands_VmbC executable from main.c, ActionCommands, Helper, and ImageAcquisition source files.
- **Library Linking** — Links the executable against Vmb::C and VmbCExamplesCommon libraries, plus pthread on Unix systems.
- **Target Properties** — Sets C11 standard and configures Visual Studio debugger environment path for VMB binaries.
