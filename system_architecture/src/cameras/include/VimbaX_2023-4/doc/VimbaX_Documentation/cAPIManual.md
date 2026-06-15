---
title: VimbaX C API Manual Documentation
description: Comprehensive HTML documentation for the Allied Vision VimbaX C API (version
  2023-4), covering camera control, image acquisition, feature access, and event handling
  for industrial cameras.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/cAPIManual.html
tags:
- VimbaX
- C API
- camera SDK
- image acquisition
- Allied Vision
related: []
last_analyzed: '2026-03-09T07:57:53Z'
---

# VimbaX C API Manual Documentation

This is the C API Manual for VimbaX 2023-4, part of the Allied Vision Vimba X Developer Guide documentation. It provides detailed guidance on using the VimbaX C API to interface with industrial cameras including GigE Vision, USB3 Vision, Camera Link, and MIPI CSI cameras. The documentation covers the complete workflow from API initialization (VmbStartup/VmbShutdown), camera discovery and enumeration, opening cameras with different access modes, accessing and modifying camera features through the GenICam feature model, image acquisition using frame buffers with asynchronous callbacks, event handling for discovery and feature invalidation notifications, chunk data access, settings persistence, and external triggering mechanisms. It includes numerous C code snippets demonstrating common operations, struct definitions (VmbCameraInfo_t, VmbFrame_t, VmbFeatureInfo_t, VmbInterfaceInfo_t), enumeration types for access modes, feature data types, and error codes.

**Key concepts:** `VmbStartup/VmbShutdown API lifecycle management`, `Camera discovery and enumeration (GigE, USB, Camera Link, MIPI CSI)`, `VmbCameraOpen/VmbCameraClose for camera access`, `Feature access API (VmbFeatureIntGet, VmbFeatureEnumSet, etc.)`, `Frame-based image capture and asynchronous acquisition`, `VmbFrame_t structure for image buffer management`, `Event notifications and callback registration`, `Settings persistence (save/load camera configurations)`, `External triggering and Action Commands`, `Interface and transport layer enumeration`, `Error codes and troubleshooting`
