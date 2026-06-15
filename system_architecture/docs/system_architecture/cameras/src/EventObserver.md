---
title: EventObserver – VmbCPP Feature Observer Implementation
description: Implements the EventObserver class, a concrete VmbCPP::IFeatureObserver
  that forwards camera feature-change notifications to a user-supplied callback function.
source: docs/system_architecture/cameras/src/EventObserver.md
tags:
- camera
- vmbcpp
- observer-pattern
- callback
- allied-vision
related: []
last_analyzed: '2026-03-09T07:42:09Z'
---

# EventObserver – VmbCPP Feature Observer Implementation

This documentation describes the EventObserver class implementation for Allied Vision Vimba C++ (VmbCPP) SDK integration. EventObserver inherits from VmbCPP::IFeatureObserver and serves as a bridge between the SDK's event system and application-level logic. It accepts a FeatureCallback on construction and stores it; when the SDK fires a feature-change event, the overridden FeatureChanged method delegates to the stored callback with the changed FeaturePtr. This design decouples SDK event wiring from application business logic and allows callers to inject arbitrary handlers at runtime.

**Key concepts:** `Observer pattern`, `VmbCPP::IFeatureObserver interface`, `Feature change notification`, `Callback delegation`, `Allied Vision Vimba SDK`

## Sections

- **EventObserver – VmbCPP Feature Observer Implementation** — Introduces the EventObserver class and its role as a bridge between VmbCPP SDK events and application callbacks.
- **Exports** — Documents the constructor that stores a FeatureCallback and the FeatureChanged override that forwards events to the callback.
- **Dependencies** — Lists VmbCPP (Allied Vision Vimba C++ SDK) as the dependency providing IFeatureObserver and FeaturePtr.
