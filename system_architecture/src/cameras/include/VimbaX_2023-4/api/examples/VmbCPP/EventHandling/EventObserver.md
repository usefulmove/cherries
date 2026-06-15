---
title: EventObserver
description: Header file defining an event observer class that monitors camera feature
  changes in the VmbCPP SDK example for VimbaX.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/EventHandling/EventObserver.h
tags:
- vimba
- camera-sdk
- event-handling
- observer-pattern
- cpp-header
related: []
last_analyzed: '2026-03-09T07:49:53Z'
---

# EventObserver

This header file defines the EventObserver class, which is part of the VimbaX SDK examples for Allied Vision cameras. The class inherits from VmbCPP::IFeatureObserver and implements the observer pattern to receive callbacks when camera features change. It declares a single virtual method FeatureChanged that will be invoked whenever an observed camera feature is modified, enabling event-driven handling of camera state changes.

**Key concepts:** `IFeatureObserver interface`, `Observer pattern`, `Camera feature change notification`, `VmbCPP SDK`, `Event-driven programming`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class EventObserver : public VmbCPP::IFeatureObserver` | class | Observer class that receives notifications when camera features change |
| `virtual void FeatureChanged(const FeaturePtr& pFeature)` | method | Callback method invoked when an observed camera feature has changed |

## Dependencies

`VmbCPP`
