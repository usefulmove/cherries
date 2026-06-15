---
title: VmbC Event Handling Header
description: Header file declaring functions for demonstrating camera event handling
  functionality in the VmbC (Vimba C) API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/EventHandling/EventHandling.h
tags:
- vmbx
- camera-events
- c-header
- allied-vision
- callback
related: []
last_analyzed: '2026-03-09T07:48:56Z'
---

# VmbC Event Handling Header

This header file is part of the VimbaX SDK examples from Allied Vision Technologies. It declares three functions for demonstrating camera event functionality: CameraEventDemo for running the main demonstration, ActivateNotification for enabling camera event notifications, and RegisterEventCallback for registering callback functions to handle camera events. The file uses VmbC API types including VmbHandle_t and VmbErrorType.

**Key concepts:** `camera event handling`, `event notifications`, `event callbacks`, `VmbC API`, `camera handle`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int CameraEventDemo(char const* cameraId)` | function | Demonstrates the camera event functionality of VmbC |
| `VmbErrorType ActivateNotification(VmbHandle_t cameraHandle)` | function | Helper function to activate camera event notifications |
| `VmbErrorType RegisterEventCallback(VmbHandle_t cameraHandle)` | function | Helper function to register an event callback function |

## Dependencies

`VmbC/VmbCommonTypes.h`
