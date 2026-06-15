---
title: VmbCPP Event Handling Example Main
description: Main entry point for the VimbaX C++ SDK event handling example, demonstrating
  camera event subscription and handling.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbCPP/EventHandling/main.cpp
tags:
- vimbax
- camera-api
- event-handling
- cpp-example
- allied-vision
related: []
last_analyzed: '2026-03-09T07:49:54Z'
---

# VmbCPP Event Handling Example Main

This is the main entry point for the VmbCPP Event Handling example from the VimbaX 2023-4 SDK by Allied Vision Technologies. The program demonstrates camera event handling using the VimbaX C++ API. It accepts an optional command-line argument for specifying a camera ID; if no camera ID is provided, it uses the first available camera. The main function creates an EventHandling instance and calls CameraEventDemo() to demonstrate event subscription and handling capabilities.

**Key concepts:** `VimbaX SDK`, `Camera event handling`, `Command-line camera selection`, `Allied Vision camera integration`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int main(int argc, char* argv[])` | function | Program entry point that parses command-line arguments and runs the camera event handling demo with an optional camera ID |
