---
title: VmbSystem - VimbaX C++ API System Singleton
description: Implementation of the VmbSystem singleton class that provides central
  management for VimbaX camera SDK, handling camera discovery, transport layers, interfaces,
  and observer pattern notifications.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/VmbSystem.cpp
tags:
- vimbax
- camera-sdk
- singleton
- observer-pattern
- device-discovery
related: []
last_analyzed: '2026-03-09T07:56:06Z'
---

# VmbSystem - VimbaX C++ API System Singleton

VmbSystem.cpp implements the core singleton class for the VimbaX C++ camera SDK by Allied Vision Technologies. It manages the lifecycle of camera connections including startup/shutdown of the underlying Vimba C API, maintains thread-safe maps of transport layers, interfaces, and cameras, and implements an observer pattern for camera and interface discovery events. The class provides methods to enumerate and access cameras by ID, open cameras with specific access modes, and register custom camera factories. It handles hot-plug events through callback functions that notify registered observers when cameras or interfaces are plugged in, unplugged, or change state. All collections are protected with condition-based read/write locks for thread safety.

**Key concepts:** `Singleton pattern for system-wide camera management`, `Transport layer abstraction for camera communication`, `Interface discovery and management`, `Camera hot-plug detection via observer callbacks`, `Thread-safe collections with read/write locks`, `Factory pattern for camera object creation`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbSystem& VmbSystem::GetInstance() noexcept` | method | Returns the singleton instance of VmbSystem |
| `VmbErrorType VmbSystem::QueryVersion(VmbVersionInfo_t &rVersion) const noexcept` | method | Queries the VmbCPP library version information |
| `VmbErrorType VmbSystem::Startup(const VmbFilePathChar_t* pathConfiguration)` | method | Initializes the Vimba API with optional configuration path |
| `VmbErrorType VmbSystem::Startup()` | method | Initializes the Vimba API with default configuration |
| `VmbErrorType VmbSystem::Shutdown()` | method | Shuts down the Vimba API, closes cameras, and clears all internal lists |
| `VmbErrorType VmbSystem::GetInterfaces(InterfacePtr *pInterfaces, VmbUint32_t &rnSize)` | method | Gets list of available interfaces or their count |
| `VmbErrorType VmbSystem::GetInterfaceByID(const char *pStrID, InterfacePtr &rInterface)` | method | Retrieves an interface by its string identifier |
| `VmbErrorType VmbSystem::GetCameras(CameraPtr *pCameras, VmbUint32_t &rnSize)` | method | Gets list of available cameras or their count |
| `VmbErrorType VmbSystem::GetCameraByID(const char *pStrID, CameraPtr &rCamera)` | method | Retrieves a camera by its string identifier (ID, serial, or IP) |
| `VmbErrorType VmbSystem::OpenCameraByID(const char *pStrID, VmbAccessModeType eAccessMode, CameraPtr &rCamera)` | method | Opens a camera by ID with specified access mode |
| `CameraPtr VmbSystem::GetCameraPtrByHandle(const VmbHandle_t handle) const` | method | Retrieves a camera pointer by its native handle |
| `VmbErrorType VmbSystem::GetTransportLayers(TransportLayerPtr *pTransportLayers, VmbUint32_t &rnSize) noexcept` | method | Gets list of available transport layers or their count |
| `VmbErrorType VmbSystem::GetTransportLayerByID(const char *pStrID, TransportLayerPtr &rTransportLayer)` | method | Retrieves a transport layer by its string identifier |
| `VmbErrorType VmbSystem::RegisterCameraListObserver(const ICameraListObserverPtr &rObserver)` | method | Registers an observer for camera plug/unplug events |
| `VmbErrorType VmbSystem::UnregisterCameraListObserver(const ICameraListObserverPtr &rObserver)` | method | Unregisters a camera list observer |
| `VmbErrorType VmbSystem::RegisterInterfaceListObserver(const IInterfaceListObserverPtr &rObserver)` | method | Registers an observer for interface discovery events |
| `VmbErrorType VmbSystem::UnregisterInterfaceListObserver(const IInterfaceListObserverPtr &rObserver)` | method | Unregisters an interface list observer |
| `VmbErrorType VmbSystem::RegisterCameraFactory(const ICameraFactoryPtr &cameraFactory)` | method | Registers a custom camera factory for creating camera objects |
| `VmbErrorType VmbSystem::UnregisterCameraFactory()` | method | Reverts to the default camera factory |
| `Logger* VmbSystem::GetLogger() const noexcept` | method | Returns the internal logger instance |
