---
title: VimbaX Python API Manual (VmbPy)
description: Comprehensive documentation for the VmbPy Python API, a wrapper around
  the VmbC API for controlling Allied Vision cameras in machine vision applications.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/pythonAPIManual.html
tags:
- vimba
- python-api
- camera-sdk
- machine-vision
- documentation
related: []
last_analyzed: '2026-03-09T07:58:46Z'
---

# VimbaX Python API Manual (VmbPy)

This HTML documentation file is part of the Vimba X Developer Guide (version 2023-4) and provides the complete Python API Manual for VmbPy. It covers installation instructions for Windows, Linux, and macOS (requiring Python 3.7+), general API concepts including the VmbSystem singleton entry point and context manager patterns, and detailed API usage including listing cameras, accessing features, image acquisition (both synchronous and asynchronous), pixel format manipulation, chunk data handling, user sets, software triggers, Action Commands for GigE cameras, multithreading considerations, and logging capabilities. The documentation recommends VmbPy for quick prototyping and getting started with machine vision applications, while noting that C/C++ APIs provide better performance for production use. Examples are referenced throughout for practical implementation guidance.

**Key concepts:** `VmbPy - Python wrapper around VmbC API`, `VmbSystem singleton as entry point`, `Context manager pattern for camera/interface access`, `Camera, Frame, and Interface classes`, `Synchronous and asynchronous image acquisition`, `Feature access and manipulation`, `Pixel format conversion`, `Software triggers and Action Commands`, `Chunk data support`, `NumPy and OpenCV integration`, `Logging with TraceEnable and ScopedLogEnable decorators`

## Sections

- **Introduction to the API** — Overview of VmbPy as a Python wrapper for VmbC, recommended use cases
- **Compatibility** — Platform and version compatibility information
- **Installation** — Prerequisites and installation instructions for Windows, Linux, and macOS
- **General aspects of the API** — Entry point, context managers, and class structures (Camera, Frame, Interface)
- **API usage** — Practical usage including camera listing, feature access, image acquisition, pixel formats, chunk data, triggers, and multithreading
- **Troubleshooting** — Common issues and solutions
- **Logging** — Logging configuration, levels, and decorators (TraceEnable, ScopedLogEnable)
