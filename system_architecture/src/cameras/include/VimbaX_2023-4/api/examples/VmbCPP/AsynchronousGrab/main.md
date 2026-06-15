---
title: VmbCPP Asynchronous Grab Example Main Entry Point
description: Main entry point for the VmbCPP asynchronous image grab example demonstrating
  camera acquisition using the Vimba X SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/AsynchronousGrab/main.cpp
tags:
- vimba-x
- camera-acquisition
- cpp-example
- async-grab
related: []
last_analyzed: '2026-03-09T07:49:28Z'
---

# VmbCPP Asynchronous Grab Example Main Entry Point

This is the main entry point for the VmbCPP Asynchronous Grab example from the Vimba X SDK. It demonstrates basic asynchronous camera image acquisition by creating an AcquisitionHelper object, starting the acquisition, and waiting for user input to stop. The example uses RAII pattern where the AcquisitionHelper destructor handles cleanup of the acquisition and API shutdown. Error handling is done via exception catching for runtime errors.

**Key concepts:** `asynchronous image acquisition`, `camera API initialization`, `RAII resource management`, `exception handling`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int main()` | function | Program entry point that initializes asynchronous camera acquisition, waits for user input, and handles cleanup via RAII |
