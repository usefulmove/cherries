---
title: VimbaX C API Configuration File
description: XML configuration file for the VimbaX C API (VmbC) that controls logging,
  transport layer loading, and device polling settings for camera interfaces.
source: src/cameras/include/VimbaX_2023-4/bin/VmbC.xml
tags:
- vimba
- camera-api
- configuration
- xml
- allied-vision
related: []
last_analyzed: '2026-03-09T07:55:26Z'
---

# VimbaX C API Configuration File

This is an XML configuration file for the Allied Vision VimbaX C API (VmbC). It provides settings for controlling API behavior including optional logging with configurable file output and append mode, transport layer (CTI) loading with customizable paths, vendor filtering (AVT), and interface type filtering (GigE Vision). The file also configures polling intervals for interface discovery (default 2000ms) and device list updates. Most settings are commented out, indicating default behavior is used. This configuration supports Allied Vision camera SDK integration.

**Key concepts:** `API logging configuration`, `Transport layer (CTI) path loading`, `Interface polling period`, `Device polling period`, `GigE Vision (GEV) interface support`, `AVT vendor transport layers`

## Sections

- **Logging Configuration** — Optional log file settings including filename and append mode
- **Transport Layer Loading** — CTI paths, vendor/interface type filters, and polling periods for interface and device discovery
