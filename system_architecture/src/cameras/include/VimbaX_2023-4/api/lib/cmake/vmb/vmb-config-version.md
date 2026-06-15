---
title: VMB CMake Package Version Configuration
description: CMake configuration file that defines version compatibility checks for
  the Vimba SDK package, validating architecture and platform suitability.
source: src/cameras/include/VimbaX_2023-4/api/lib/cmake/vmb/vmb-config-version.cmake
tags:
- cmake
- vimba-sdk
- package-config
- version-check
- cross-platform
related: []
last_analyzed: '2026-03-09T07:53:15Z'
---

# VMB CMake Package Version Configuration

This is a CMake package version configuration file for the VimbaX SDK (version 1.0.5). It provides version compatibility checking logic for CMake's find_package() mechanism. The file validates system architecture compatibility, requiring x86_64 architecture on 64-bit Linux systems. It includes platform-specific checks for Apple/macOS, Linux (x86_64-linux-gnu), and Windows systems. The configuration sets PACKAGE_VERSION_COMPATIBLE and PACKAGE_VERSION_EXACT flags based on whether the requested version matches the available version 1.0.5.

**Key concepts:** `CMake package version checking`, `Architecture compatibility (x86_64)`, `Cross-platform support (Linux, Windows, macOS)`, `Package version 1.0.5`, `64-bit system requirement`
