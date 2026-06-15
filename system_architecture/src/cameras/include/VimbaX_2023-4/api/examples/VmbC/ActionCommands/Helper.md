---
title: VmbC ActionCommands Helper Header
description: Header file declaring helper functions for the VimbaX Action Commands
  example, providing camera discovery and API initialization utilities.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ActionCommands/Helper.h
tags:
- vimbax
- camera-api
- gige-vision
- helper
- c-header
related: []
last_analyzed: '2026-03-09T07:46:28Z'
---

# VmbC ActionCommands Helper Header

This C header file is part of the VimbaX SDK's Action Commands example. It declares two helper functions: FindCamera() which searches for a compatible camera (optionally requiring an Allied Vision GigE Transport Layer), and StartApi() which initializes the VimbaX API and prints version information. These utilities support the Action Commands example by abstracting camera discovery and API startup logic.

**Key concepts:** `camera discovery`, `VimbaX API initialization`, `Action Commands`, `GigE Transport Layer`, `Allied Vision cameras`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbError_t FindCamera(const VmbBool_t needsAvtGigETL, const char* const pCameraId, VmbCameraInfo_t* const pCameraInfo)` | function | Searches for a camera that can be used by the Action Commands example, optionally requiring an Allied Vision GigE Transport Layer. Returns camera information in the provided struct. |
| `VmbError_t StartApi(void)` | function | Starts the VimbaX API and prints version information about the API. |

## Dependencies

`VmbC/VmbCTypeDefinitions.h`
