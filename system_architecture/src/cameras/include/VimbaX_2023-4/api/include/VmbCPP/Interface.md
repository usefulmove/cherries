---
title: VmbCPP Interface Class
description: Defines the VmbCPP::Interface class representing a GenTL interface for
  camera communication in the Allied Vision VimbaX SDK.
source: src/cameras/include/VimbaX_2023-4/api/include/VmbCPP/Interface.h
tags:
- vimbax-sdk
- camera-interface
- gentl
- cpp-header
- allied-vision
related: []
last_analyzed: '2026-03-09T07:52:05Z'
---

# VmbCPP Interface Class

This header file defines the VmbCPP::Interface class, which represents a GenTL interface in the Allied Vision VimbaX SDK. The Interface class inherits from PersistableFeatureContainer and provides methods to retrieve interface properties (ID, name, type), access the parent transport layer, and enumerate connected cameras. The class is non-copyable and uses the PIMPL idiom with a private Impl struct. It includes both public std::string-based methods and private IMEXPORT methods for passing data safely across DLL boundaries.

**Key concepts:** `GenTL interface`, `Camera enumeration`, `Transport layer abstraction`, `DLL boundary handling`, `Feature container pattern`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class Interface : public PersistableFeatureContainer` | class | Represents a GenTL interface for camera communication, providing access to interface properties and connected cameras |
| `using CameraPtrVector = std::vector<CameraPtr>` | type | Type alias for a vector of camera shared pointers |
| `using GetCamerasByInterfaceFunction = std::function<VmbErrorType(const Interface* pInterface, CameraPtr* pCameras, VmbUint32_t& size)>` | type | Function type for retrieving an interface's cameras |
| `Interface(const VmbInterfaceInfo_t& interfaceInfo, const TransportLayerPtr& pTransportLayerPtr, GetCamerasByInterfaceFunction getCamerasByInterface)` | method | Constructor that creates an interface given interface info and related object information |
| `virtual ~Interface()` | method | Virtual destructor |
| `VmbErrorType GetID(std::string &interfaceID) const noexcept` | method | Gets the ID of the interface |
| `IMEXPORT VmbErrorType GetType(VmbTransportLayerType& type) const noexcept` | method | Gets the type of interface (e.g., GigE or USB) |
| `VmbErrorType GetName(std::string& name) const noexcept` | method | Gets the name of the interface |
| `IMEXPORT VmbErrorType GetTransportLayer(TransportLayerPtr& pTransportLayer) const` | method | Gets the pointer of the related transport layer |
| `VmbErrorType GetCameras(CameraPtrVector& cameras)` | method | Gets all cameras related to this interface |

## Dependencies

`VmbC/VmbC.h`
