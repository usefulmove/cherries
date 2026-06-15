---
title: AccessModeToString Header
description: Header file declaring a utility function to convert VmbAccessMode_t values
  to human-readable strings for VimbaX camera SDK examples.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/include/VmbCExamplesCommon/AccessModeToString.h
tags:
- vimbax
- camera-sdk
- utility
- c-header
related: []
last_analyzed: '2026-03-09T07:48:20Z'
---

# AccessModeToString Header

This is a C header file that declares a utility function `AccessModesToString` for the VimbaX camera SDK examples. The function converts VmbAccessMode_t enumeration values (representing camera access modes) into human-readable string representations. This is part of the common utilities shared across VmbC example applications in the Allied Vision VimbaX 2023-4 SDK.

**Key concepts:** `VmbAccessMode_t`, `access mode conversion`, `string representation`, `VimbaX SDK utilities`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `const char* AccessModesToString(VmbAccessMode_t eMode)` | function | Translates Vmb access modes to a readable string representation |

## Dependencies

`VmbC/VmbCTypeDefinitions.h`
