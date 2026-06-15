---
title: VimbaX GenTL Path Configuration Script
description: Shell script that sets GENICAM_GENTL64_PATH environment variable for
  VimbaX camera applications running in non-login user contexts like services.
source: src/cameras/include/VimbaX_2023-4/cti/SetGenTLPath.sh
tags:
- shell-script
- camera-configuration
- environment-setup
- genicam
- vimbax
related: []
last_analyzed: '2026-03-09T07:55:37Z'
---

# VimbaX GenTL Path Configuration Script

This shell script is part of the VimbaX 2023-4 SDK from Allied Vision Technologies and is used to configure the GenICam Transport Layer (GenTL) path for camera applications. It detects the system architecture (ARM64 or x86_64), validates that it's being executed via the 'source' command (required to affect the current shell environment), and exports the GENICAM_GENTL64_PATH environment variable. This is particularly useful when running VimbaX applications under non-logged-in users such as system services.

**Key concepts:** `GenTL path configuration`, `GENICAM_GENTL64_PATH environment variable`, `Architecture detection (aarch64/x86_64)`, `Source script execution pattern`, `Allied Vision camera SDK setup`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `export GENICAM_GENTL64_PATH=:$TL_PATH_64BIT` | constant | Environment variable set to the directory containing the GenTL producer (.cti) files for 64-bit systems |
