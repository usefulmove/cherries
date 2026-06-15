---
title: VimbaX VmbC API Configuration File
description: XML configuration file for the Vimba X camera SDK's VmbC API, containing
  settings for logging, transport layer loading, and interface polling.
source: src/cameras/include/VimbaX_2023-4/api/lib/VmbC.xml
tags:
- vimba
- camera-sdk
- configuration
- xml
- api-settings
related: []
last_analyzed: '2026-03-09T07:52:50Z'
---

# VimbaX VmbC API Configuration File

This XML configuration file defines settings for the Allied Vision VimbaX SDK's VmbC (C API) component. It includes options for enabling/disabling logging with configurable log file paths, transport layer (TL) loading configuration including CTI (Camera Transport Interface) paths, vendor and interface type filtering (supporting AVT vendor and GEV/GigE Vision interfaces), interface polling period (set to 2000ms by default), and device polling/discovery timeout settings. Most options are commented out with documentation explaining default behaviors.

**Key concepts:** `API logging configuration`, `Transport Layer (TL) loading settings`, `CTI paths configuration`, `Interface polling period`, `Device polling settings`, `GenICam/GigE Vision camera interface`

## Sections

- **LogFileName** — Configuration for enabling logging and setting the log file path
- **AppendLog** — Configuration for log file append vs reset behavior
- **TlLoading** — Transport layer loading configuration including CTI paths, vendor filtering, interface types, and polling settings
