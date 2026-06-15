---
title: ICameraFactory Interface
description: Defines the abstract factory interface for creating custom Camera objects
  in the VmbCPP (Vimba C++) SDK.
source: src/cameras/include/VimbaX_2023-4/api/include/VmbCPP/ICameraFactory.h
tags:
- vimba-sdk
- camera
- factory-pattern
- interface
- cpp
related: []
last_analyzed: '2026-03-09T07:51:41Z'
---

# ICameraFactory Interface

This header file defines the ICameraFactory interface, which is part of the Allied Vision Vimba C++ SDK (VmbCPP). The interface provides an abstract factory pattern that allows users to customize the creation of Camera objects by implementing the CreateCamera method. This factory can be registered with the VmbSystem before API startup to create custom Camera subclasses. The interface includes a pure virtual CreateCamera method that takes camera info and interface pointer parameters, and a virtual destructor for proper cleanup.

**Key concepts:** `abstract factory pattern`, `camera object creation`, `interface-based customization`, `VmbCPP SDK architecture`, `smart pointer usage`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class ICameraFactory` | class | Abstract factory interface for creating Camera objects in the VmbCPP SDK. Can be subclassed to customize camera object creation. |
| `IMEXPORT virtual CameraPtr CreateCamera(const VmbCameraInfo_t& cameraInfo, const InterfacePtr& pInterface) = 0` | method | Pure virtual factory method to create a Camera instance. Takes camera info struct containing ID (IP address, MAC, serial number, or device ID) and shared pointer to the connected interface. |
| `IMEXPORT virtual ~ICameraFactory() {}` | method | Virtual destructor to ensure proper cleanup of derived factory implementations. |

## Dependencies

`VmbC/VmbC.h`, `VmbCPP/VmbCPPCommon.h`, `VmbCPP/SharedPointerDefines.h`, `VmbCPP/Interface.h`, `VmbCPP/Camera.h`
