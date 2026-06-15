---
title: ConfigIp VmbC Example CMakeLists.txt
description: CMake build configuration file for the ConfigIp example application in
  the VimbaX SDK, which demonstrates IP configuration for GigE Vision cameras using
  the VmbC API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ConfigIp/CMakeLists.txt
tags:
- cmake
- vimbax-sdk
- camera-configuration
- build-system
- gige-vision
related: []
last_analyzed: '2026-03-09T07:48:44Z'
---

# ConfigIp VmbC Example CMakeLists.txt

This CMakeLists.txt file defines the build configuration for the ConfigIp example application, part of the VimbaX 2023-4 SDK. The ConfigIp utility allows configuration of IP addresses for GigE Vision cameras. The build system requires CMake 3.0+, links against the VmbC API and common example utilities, and includes platform-specific handling for Windows (Ws2_32 socket library). The project is configured with C11 standard and includes Visual Studio debugger environment setup for proper DLL resolution.

**Key concepts:** `CMake build configuration`, `VimbaX VmbC API`, `ConfigIp example application`, `Cross-platform build support`, `Network socket libraries for Windows`

## Sections

- **Project Definition** — Defines the ConfigIp project as a C language project requiring CMake 3.0 minimum.
- **Package Configuration** — Includes optional hardcoded package location paths and finds the required Vmb package with C components.
- **Common Dependencies** — Adds the VmbCExamplesCommon subdirectory if it's not already a target.
- **Executable Target** — Creates the ConfigIp_VmbC executable from main.c, ConfigIp.c/h, ConfigIpProg.c/h, and common sources.
- **Build Configuration** — Sets up compile definitions, library linking (including Windows-specific Ws2_32), and target properties for C11 standard and VS debugger environment.
