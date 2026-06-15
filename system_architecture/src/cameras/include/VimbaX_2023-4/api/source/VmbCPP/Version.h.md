---
title: VmbCPP Version Header Template
description: A CMake configure file template that generates version number definitions
  for the VmbCPP (Vimba C++ API) library.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/Version.h.in
tags:
- vimbax
- cmake-template
- version-header
- allied-vision
- camera-sdk
related: []
last_analyzed: '2026-03-09T07:55:20Z'
---

# VmbCPP Version Header Template

This is a CMake template file (.h.in) for the VmbCPP (Vimba C++ API) library from Allied Vision Technologies' VimbaX SDK. During the build process, CMake's configure_file command substitutes the placeholder variables ${VMB_MAJOR_VERSION}, ${VMB_MINOR_VERSION}, and ${VMB_PATCH_VERSION} with actual version numbers to generate a Version.h header file. This header defines VMBCPP_VERSION_MAJOR, VMBCPP_VERSION_MINOR, and VMBCPP_VERSION_PATCH macros that can be used throughout the codebase for version checking and compatibility.

**Key concepts:** `CMake configure_file template`, `Version number macros (MAJOR, MINOR, PATCH)`, `VmbCPP library versioning`, `Header include guard pattern`, `Build-time variable substitution`

## Sections

- **Copyright Header** — Allied Vision Technologies copyright notice and file metadata
- **Version Definitions** — Include guard and version macro definitions with CMake variable placeholders
