---
title: VmbCPP Git Version Header
description: Header file defining the VmbCPP version tweak constant for git-based
  versioning in the VimbaX camera SDK.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/Version_git.h
tags:
- version
- vimbax
- camera-sdk
- header
related: []
last_analyzed: '2026-03-09T07:55:20Z'
---

# VmbCPP Git Version Header

This is a minimal C/C++ header file that defines a version tweak macro (VMBCPP_VERSION_TWEAK) set to 0 for the VimbaX camera SDK's C++ API. It uses standard include guards to prevent multiple inclusion. This file is typically auto-generated or maintained to track git-based version information for the VmbCPP library component of the Allied Vision VimbaX SDK.

**Key concepts:** `version-control`, `preprocessor-macros`, `include-guards`, `build-versioning`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `#define VMBCPP_VERSION_TWEAK 0` | constant | Preprocessor macro defining the version tweak number (patch/build number) for VmbCPP, set to 0 |
