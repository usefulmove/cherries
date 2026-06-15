---
title: VimbaX Image Transform Library Manual
description: Comprehensive documentation for the VimbaX Image Transform library, covering
  pixel format conversions, debayering, color correction, and image transformation
  APIs for C/C++ development.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/imagetransformManual.html
tags:
- image-processing
- vimba-x
- pixel-format
- debayering
- camera-sdk
related: []
last_analyzed: '2026-03-09T07:58:06Z'
---

# VimbaX Image Transform Library Manual

This HTML documentation file is the Image Transform Manual for VimbaX 2023-4, part of the Vimba X Developer Guide. It provides detailed guidance on using the Image Transform library to convert camera images between various pixel formats without requiring knowledge of GenICam's PFNC naming scheme. The manual covers library variants (standard single-threaded and OpenMP multi-threaded), supported image data formats including Mono, Bayer, RGB, BGR, RGBA, and YUV formats at various bit depths (8, 10, 12, 14, 16-bit). It includes comprehensive API usage documentation with the main VmbImageTransform() function, helper functions like VmbSetImageInfoFromPixelFormat() and VmbSetImageInfoFromInputImage(), and struct definitions (VmbImage, VmbImageInfo, VmbPixelInfo, VmbTransformInfo). The manual provides practical C++ code examples for debayering, color correction, and advanced transformations, along with performance optimization tips for OpenMP.

**Key concepts:** `Image transformation between pixel formats`, `Debayering algorithms (2x2, 3x3)`, `PFNC (Pixel Format Naming Convention)`, `OpenMP parallel processing variant`, `VmbImage and VmbImageInfo structs`, `Color correction and gamma correction`, `Supported pixel formats (Mono, Bayer, RGB, BGR, RGBA, YUV)`, `VmbImageTransform() main transformation function`, `Helper functions for image info setup`
