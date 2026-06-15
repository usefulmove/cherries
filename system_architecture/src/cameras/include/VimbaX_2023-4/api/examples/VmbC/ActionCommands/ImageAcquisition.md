---
title: VmbC ImageAcquisition Header
description: Header file declaring functions for starting and stopping camera image
  streaming in the VimbaX Action Commands example.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ActionCommands/ImageAcquisition.h
tags:
- vimbax
- camera-streaming
- image-acquisition
- c-api
- header
related: []
last_analyzed: '2026-03-09T07:46:32Z'
---

# VmbC ImageAcquisition Header

This header file is part of the VimbaX SDK's VmbC Action Commands example. It declares two functions for controlling camera image streaming: StartStream() to prepare and start image acquisition from an opened camera, and StopStream() to stop and clean up the streaming session. Both functions take a camera handle parameter and return VmbError_t for error handling. The file uses the VmbC type definitions from the VimbaX SDK.

**Key concepts:** `camera streaming`, `VmbC API`, `action commands`, `image acquisition`, `stream control`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbError_t StartStream(const VmbHandle_t cameraHandle)` | function | Prepares and starts the image stream for an already opened camera |
| `VmbError_t StopStream(const VmbHandle_t cameraHandle)` | function | Stops the stream and reverts the steps done during StartStream |

## Dependencies

`VmbC/VmbCTypeDefinitions.h`
