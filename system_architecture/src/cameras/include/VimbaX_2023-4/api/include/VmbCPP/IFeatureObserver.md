---
title: IFeatureObserver Interface
description: Defines an abstract base class for feature change observers in the VmbCPP
  (Vimba C++ API), enabling event-driven notifications when camera features change.
source: src/cameras/include/VimbaX_2023-4/api/include/VmbCPP/IFeatureObserver.h
tags:
- vimba
- camera-api
- observer-pattern
- interface
- cpp
related: []
last_analyzed: '2026-03-09T07:51:57Z'
---

# IFeatureObserver Interface

This header file defines the IFeatureObserver interface for the VmbCPP (Vimba C++ API) from Allied Vision Technologies. It provides an abstract base class that users must derive from to receive notifications when camera features change. The interface follows the observer pattern, requiring derived classes to implement the pure virtual FeatureChanged() method which receives a FeaturePtr parameter indicating which feature has changed. The class includes protected constructors (default and copy) and a copy assignment operator for use by derived classes, as well as a virtual destructor for proper cleanup. The IMEXPORT macro is used to control symbol visibility for DLL export/import.

**Key concepts:** `Observer design pattern`, `Feature change notification`, `Abstract base class interface`, `Event-driven programming`, `Allied Vision camera SDK`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class IFeatureObserver` | class | Abstract base class for feature invalidation listeners. Derived classes must implement FeatureChanged() to receive feature change notifications. |
| `IMEXPORT virtual void FeatureChanged( const FeaturePtr &pFeature ) = 0` | method | Pure virtual event handler function called whenever a feature has changed. Must be implemented by derived classes. |
| `IMEXPORT virtual ~IFeatureObserver()` | method | Virtual destructor for proper cleanup of derived classes. |
| `IMEXPORT IFeatureObserver()` | method | Protected default constructor for use by derived classes. |
| `IMEXPORT IFeatureObserver( const IFeatureObserver& )` | method | Protected copy constructor for use by derived classes. |
| `IMEXPORT IFeatureObserver& operator=( const IFeatureObserver& )` | method | Protected copy assignment operator for use by derived classes. |
