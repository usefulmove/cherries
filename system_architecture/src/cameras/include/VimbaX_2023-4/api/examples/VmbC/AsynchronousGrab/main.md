---
title: VmbC Asynchronous Grab Example Main Entry Point
description: Main entry point for the Allied Vision VmbC API asynchronous image acquisition
  example, handling command-line parsing and camera acquisition lifecycle.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/AsynchronousGrab/main.c
tags:
- vimba-sdk
- camera-acquisition
- command-line-parser
- example-code
- allied-vision
related: []
last_analyzed: '2026-03-09T07:46:58Z'
---

# VmbC Asynchronous Grab Example Main Entry Point

This file is the main entry point for the Allied Vision VmbC Asynchronous Grab example application. It parses command-line arguments to configure camera acquisition options (camera ID, RGB conversion, color processing, frame info display, and buffer allocation mode), handles Windows console signals for graceful shutdown, and orchestrates the continuous image acquisition lifecycle using StartContinuousImageAcquisition and StopContinuousImageAcquisition functions. After acquisition stops, it displays frame statistics including complete, incomplete, too small, invalid, and missing frame counts.

**Key concepts:** `asynchronous image acquisition`, `command-line argument parsing`, `console signal handling (Windows)`, `frame statistics tracking`, `VmbC API usage`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `BOOL WINAPI ConsoleHandler(DWORD signal)` | function | Windows-specific console signal handler that stops image acquisition on CTRL_C or CTRL_CLOSE events |
| `void PrintUsage(void)` | function | Prints command-line usage information showing available parameters and their descriptions |
| `VmbError_t ParseCommandLineParameters(AsynchronousGrabOptions* cmdOptions, VmbBool_t* printHelp, int argc, char* argv[])` | function | Parses command-line arguments into AsynchronousGrabOptions struct, supporting camera ID, RGB display, color processing, frame info modes, and alloc-and-announce buffer mode |
| `int main(int argc, char* argv[])` | function | Application entry point that initializes options, starts/stops continuous image acquisition, and reports frame statistics |

## Dependencies

`VmbC/VmbC.h`, `AsynchronousGrab.h`, `VmbCExamplesCommon/ErrorCodeToMessage.h`
