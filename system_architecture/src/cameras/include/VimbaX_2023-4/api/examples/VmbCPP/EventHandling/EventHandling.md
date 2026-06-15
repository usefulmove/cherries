---
title: EventHandling VmbCPP Example Header
description: Header file defining the EventHandling class that demonstrates camera
  event functionality using the VimbaX C++ API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/EventHandling/EventHandling.h
tags:
- vimbax
- camera-api
- event-handling
- cpp-header
- allied-vision
related: []
last_analyzed: '2026-03-09T07:49:51Z'
---

# EventHandling VmbCPP Example Header

This header file defines the EventHandling class within the VmbCPP::Examples namespace, which serves as an example demonstrating camera event functionality using Allied Vision's VimbaX C++ API. The class provides a public CameraEventDemo method that accepts a camera ID string, and private helper methods for activating event notifications and registering feature observers. It maintains a CameraPtr member and defines an acquisition timeout constant of 2000 milliseconds.

**Key concepts:** `camera event notifications`, `feature observer pattern`, `VimbaX C++ API integration`, `camera event demonstration`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class EventHandling` | class | Example class demonstrating VimbaX camera event handling functionality |
| `void CameraEventDemo(std::string cameraID)` | method | Public method to demonstrate the camera event functionality of VmbCPP |
| `VmbErrorType ActivateNotification()` | method | Private helper function to activate camera event notifications |
| `VmbErrorType RegisterEventObserver()` | method | Private helper function to create and register a feature observer |
| `CameraPtr m_Camera` | other | Private member holding the camera pointer |
| `static const VmbInt32_t k_AquisitionTimeout = 2000` | constant | Private static constant defining the acquisition timeout in milliseconds |

## Dependencies

`VmbCPP`
