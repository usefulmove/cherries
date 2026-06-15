---
title: ListCameras Example Header
description: Header file for a VmbCPP example class that prints information about
  connected cameras using the Vimba X SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/ListCameras/ListCameras.h
tags:
- vimba-x
- camera-api
- cpp-header
- example-code
- allied-vision
related: []
last_analyzed: '2026-03-09T07:50:01Z'
---

# ListCameras Example Header

This header file defines the ListCameras class, part of the VmbCPP Examples namespace within the Vimba X 2023-4 SDK from Allied Vision Technologies. The class provides a single static method Print() that initializes the Vimba system, discovers all connected cameras, and outputs their details including camera name, model name, serial number, camera ID, and interface ID. This serves as an example implementation for camera enumeration using the VmbCPP API.

**Key concepts:** `camera enumeration`, `Vimba X SDK`, `camera information display`, `static class method pattern`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class ListCameras` | class | Example class demonstrating camera enumeration with the Vimba X SDK |
| `static void Print()` | method | Starts Vimba, gets all connected cameras, and prints camera name, model name, serial number, ID, and interface ID |
