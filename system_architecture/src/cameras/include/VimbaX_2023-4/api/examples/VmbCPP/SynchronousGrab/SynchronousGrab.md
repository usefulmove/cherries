---
title: SynchronousGrab Camera Acquisition Example
description: Example implementation demonstrating synchronous single image acquisition
  from Allied Vision cameras using the VmbCPP (Vimba X) API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/SynchronousGrab/SynchronousGrab.cpp
tags:
- camera
- vimba
- image-acquisition
- gige-vision
- allied-vision
related: []
last_analyzed: '2026-03-09T07:50:11Z'
---

# SynchronousGrab Camera Acquisition Example

This file implements the SynchronousGrab class for Allied Vision's VmbCPP (Vimba X) SDK. It provides functionality to initialize the Vimba system, discover and open cameras (either by ID or first available), adjust GigE packet size for optimal performance, and acquire single images synchronously with a 5-second timeout. The class manages the camera lifecycle including proper startup and shutdown of the Vimba system, with comprehensive error handling that throws runtime exceptions on failure.

**Key concepts:** `synchronous image capture`, `VmbCPP API usage`, `GigE camera packet size adjustment`, `camera initialization and shutdown`, `single frame acquisition`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `SynchronousGrab::SynchronousGrab()` | method | Default constructor that delegates to the parameterized constructor with nullptr, selecting the first available camera. |
| `SynchronousGrab::SynchronousGrab(const char* cameraId)` | method | Parameterized constructor that initializes the Vimba system, discovers cameras, opens a camera by ID (or first available if nullptr), and adjusts GigE packet size. |
| `SynchronousGrab::~SynchronousGrab()` | method | Destructor that shuts down the Vimba system cleanly. |
| `void SynchronousGrab::AcquireImage()` | method | Acquires a single image synchronously from the camera with a 5000ms timeout, storing result in the frame member variable. |
| `void GigEAdjustPacketSize(CameraPtr camera)` | function | Helper function that adjusts the GVSP packet size for Allied Vision GigE cameras by running the GVSPAdjustPacketSize command on the camera's stream. |

## Dependencies

`VmbCPP`
