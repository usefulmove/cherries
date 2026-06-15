---
title: ForceIp VmbC Example CMake Configuration
description: CMake build configuration for the ForceIp example application that demonstrates
  IP address forcing functionality using the VimbaX C API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ForceIp/CMakeLists.txt
tags:
- cmake
- vimbax
- force-ip
- camera-api
- build-configuration
related: []
last_analyzed: '2026-03-09T07:49:01Z'
---

# ForceIp VmbC Example CMake Configuration

This CMakeLists.txt file configures the build process for the ForceIp example application, which is part of the VimbaX 2023-4 SDK examples. The configuration sets up a C project using CMake 3.0+, finds the required Vmb package components, builds an executable from multiple source files (main.c, ForceIp.c, ForceIpProg.c), and links against the VimbaX C library and common examples library. It includes platform-specific handling for Windows (linking Ws2_32 for network functionality) and sets C11 as the language standard.

**Key concepts:** `CMake build system`, `VimbaX C API integration`, `ForceIp example application`, `Cross-platform build configuration`, `Target properties and linking`

## Sections

- **CMake Project Setup** — Defines the minimum CMake version (3.0) and creates a C-language project named ForceIp.
- **Package Discovery** — Conditionally includes hardcoded package paths and finds the required Vmb package with C components.
- **Common Library Inclusion** — Adds the VmbCExamplesCommon subdirectory if not already included as a target.
- **Executable Definition** — Creates the ForceIp_VmbC executable from main.c, ForceIp.c, ForceIp.h, ForceIpProg.c, ForceIpProg.h, and common sources.
- **Compilation and Linking** — Sets compile definitions for little-endian architecture and links against Vmb::C, VmbCExamplesCommon, and Ws2_32 on Windows.
- **Target Properties** — Configures the target to use C11 standard and sets Visual Studio debugger environment path.
