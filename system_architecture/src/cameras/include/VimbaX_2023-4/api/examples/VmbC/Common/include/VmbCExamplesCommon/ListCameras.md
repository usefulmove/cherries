---
title: ListCameras Header
description: Header file declaring a function to retrieve a list of available cameras
  using the VmbC API from Allied Vision's VimbaX SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/include/VmbCExamplesCommon/ListCameras.h
tags:
- vimbax
- camera-api
- c-header
- allied-vision
- vmbc
related: []
last_analyzed: '2026-03-09T07:48:29Z'
---

# ListCameras Header

This header file is part of the VimbaX SDK examples from Allied Vision Technologies. It declares the ListCameras function which retrieves a list of available cameras connected to the system. The function allocates memory using malloc and returns camera information through the VmbCameraInfo_t structure. It uses VimbaX/VmbC type definitions for error handling and data structures.

**Key concepts:** `camera enumeration`, `VmbC API`, `camera info retrieval`, `memory allocation for camera list`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbError_t ListCameras(VmbCameraInfo_t** cameras, VmbUint32_t* count)` | function | Gets a list of cameras, allocating an array of VmbCameraInfo_t structures using malloc. Returns VmbErrorNotFound if no cameras are found instead of setting count to 0. |

## Dependencies

`VmbC/VmbCTypeDefinitions.h`
