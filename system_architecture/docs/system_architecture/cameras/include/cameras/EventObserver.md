---
title: EventObserver – VmbCPP Feature Change Observer
description: Declares the EventObserver class, a concrete implementation of VmbCPP::IFeatureObserver
  that dispatches camera feature-change events to a user-supplied callback.
source: docs/system_architecture/cameras/include/cameras/EventObserver.md
tags:
- camera
- observer-pattern
- vmbcpp
- event-handling
- callback
related: []
last_analyzed: '2026-03-09T07:42:00Z'
---

# EventObserver – VmbCPP Feature Change Observer

This document describes the EventObserver class defined in EventObserver.h, which provides a bridge between Allied Vision's Vimba C++ SDK event notification system and application-specific handling logic. EventObserver inherits from VmbCPP::IFeatureObserver and wraps a user-provided std::function callback (FeatureCallback). When the observed camera feature changes, the SDK invokes the overridden FeatureChanged method, which in turn calls the stored callback with the updated FeaturePtr. This design pattern decouples the SDK's internals from application code, allowing callers to subscribe to feature-change events using lambdas, free functions, or bound member functions.

**Key concepts:** `Observer pattern`, `VmbCPP IFeatureObserver interface`, `Feature change notification`, `std::function callback`, `Allied Vision Vimba SDK`

## Sections

- **EventObserver – VmbCPP Feature Change Observer** — Introduces the EventObserver class as a concrete VmbCPP::IFeatureObserver implementation that forwards camera feature-change events to user-supplied callbacks.
- **Exports** — Lists the exported symbols including the EventObserver class, FeatureCallback typedef, constructor, and the FeatureChanged method override.
- **Dependencies** — Specifies VmbCPP (Vimba C++ SDK) as the external dependency required by this class.
