---
title: VimbaX C API CMake Import Configuration
description: CMake configuration file for importing the VimbaX C API shared library
  target (Vmb::C) into CMake projects.
source: src/cameras/include/VimbaX_2023-4/api/lib/cmake/vmb/configs/vmb_c.cmake
tags:
- cmake
- vimbax
- camera-api
- build-configuration
- allied-vision
related: []
last_analyzed: '2026-03-09T07:53:07Z'
---

# VimbaX C API CMake Import Configuration

This is an auto-generated CMake configuration file for the VimbaX 2023-4 SDK's C API. It defines the imported target 'Vmb::C' as a shared library and sets up its include directories. The file includes protection against multiple inclusion of the same target, computes the installation prefix relative to the file location, and loads additional configuration-specific files (vmb_c-*.cmake) for different build configurations. It also verifies that all referenced files exist to ensure a valid installation.

**Key concepts:** `CMake imported target`, `Vmb::C shared library`, `VimbaX 2023-4 SDK`, `Target import protection`, `Configuration file inclusion`

## Sections

- **CMake Version Check** — Ensures CMake version 2.6 or higher is used
- **Target Protection** — Guards against multiple inclusion of the Vmb::C target
- **Import Prefix Computation** — Computes installation prefix relative to this file's location
- **Target Definition** — Creates the Vmb::C imported shared library target with include directories
- **Configuration Loading** — Loads configuration-specific cmake files (vmb_c-*.cmake)
- **File Verification** — Verifies that all imported target files exist
