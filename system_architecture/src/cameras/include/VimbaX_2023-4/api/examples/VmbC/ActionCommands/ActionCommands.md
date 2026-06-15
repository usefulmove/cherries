---
title: VimbaX ActionCommands API Header
description: Header file defining the interface for sending GigE Vision Action Commands
  to trigger camera image acquisition using the VimbaX SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ActionCommands/ActionCommands.h
tags:
- vimbax
- camera-api
- action-commands
- gige-vision
- c-header
related: []
last_analyzed: '2026-03-09T07:46:29Z'
---

# VimbaX ActionCommands API Header

This header file is part of the VimbaX SDK examples demonstrating how to use GigE Vision Action Commands to trigger cameras. It defines the ActionCommandsOptions struct that holds configuration parameters (device key, group key, group mask, camera ID, and transmission mode flags), and declares four functions: SendActionCommand for acquiring images via action commands, PrepareCameraForActionCommands for configuring cameras to respond to action triggers, PrepareActionCommand for setting up broadcast action commands, and PrepareActionCommandAsUnicast for directing commands to specific cameras. The API uses VmbError_t return codes for error handling and integrates with VimbaX core types.

**Key concepts:** `Action Commands`, `GigE Vision camera control`, `Camera triggering`, `Broadcast vs unicast transmission`, `VimbaX SDK integration`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `typedef struct ActionCommandsOptions { VmbBool_t useAllInterfaces; VmbBool_t sendAsUnicast; char const* pCameraId; const VmbUint32_t deviceKey; const VmbUint32_t groupKey; const VmbUint32_t groupMask; }` | type | Helper struct containing command line options for configuring Action Commands, including interface selection, unicast mode, camera ID, and the device/group keys and mask. |
| `VmbError_t SendActionCommand(const ActionCommandsOptions* const pOptions, const VmbCameraInfo_t* const pCamera)` | function | Sends an Action Command to acquire and grab an image from the specified camera. |
| `VmbError_t PrepareCameraForActionCommands(const VmbHandle_t camera)` | function | Configures an already-opened camera to be triggered by Action Commands. |
| `VmbError_t PrepareActionCommand(const VmbHandle_t handle, const ActionCommandsOptions* const pOptions)` | function | Configures Action Command features on a Transport Layer or Interface handle for broadcast transmission. |
| `VmbError_t PrepareActionCommandAsUnicast(const VmbHandle_t handle, const ActionCommandsOptions* const pOptions, const VmbCameraInfo_t* const pCameraInfo)` | function | Configures Action Command features for unicast transmission directly to a specific camera. |

## Dependencies

`VmbC/VmbCTypeDefinitions.h`
