---
title: VimbaX C++ API CMake Import Configuration
description: CMake-generated configuration file for importing the Vmb::CPP shared
  library target from the VimbaX 2023-4 camera SDK.
source: src/cameras/include/VimbaX_2023-4/api/lib/cmake/vmb/configs/vmb_cpp.cmake
tags:
- cmake
- vimbax
- camera-sdk
- cpp-api
- build-configuration
related: []
last_analyzed: '2026-03-09T07:53:05Z'
---

# VimbaX C++ API CMake Import Configuration

This is a CMake-generated configuration file for the VimbaX 2023-4 camera SDK's C++ API. It defines the imported target 'Vmb::CPP' as a shared library, specifying C++11 as the required compile feature, setting up include directories, and establishing a dependency on the 'Vmb::C' target. The file includes protection against multiple inclusions, computes the installation prefix relative to the file location, loads configuration-specific files (vmb_cpp-*.cmake), and validates that all referenced files and dependent targets exist. It requires CMake version 2.8.12 or greater.

**Key concepts:** `CMake imported target (Vmb::CPP)`, `Shared library configuration`, `C++11 standard requirement`, `Dependency on Vmb::C target`, `Target existence verification`, `Import prefix computation`

## Dependencies

`Vmb::C`

## Sections

- **Version and policy setup** — 
- **Multiple inclusion protection** — 
- **Import prefix computation** — 
- **Vmb::CPP target definition** — 
- **Configuration file loading** — 
- **File existence verification** — 
- **Dependency target verification** — 
