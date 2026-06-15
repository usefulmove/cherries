---
title: VimbaX Supported Image Format Transformations Chart
description: An SVG diagram displaying a matrix of supported pixel format transformations
  in the VimbaX camera SDK, showing which source formats can be converted to which
  destination formats.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_images/Supported-transformations.svg
tags:
- VimbaX
- image-format-conversion
- pixel-formats
- camera-SDK
- documentation-diagram
related: []
last_analyzed: '2026-03-09T07:56:35Z'
---

# VimbaX Supported Image Format Transformations Chart

This SVG file is a visual reference chart from the VimbaX 2023-4 documentation that illustrates supported image format transformations. The diagram is structured as a matrix table with source formats listed vertically (left column) and destination formats listed horizontally (top row). Format categories include monochrome formats (Mono8 through Mono16), color formats (Color8 through Color16), Bayer pattern formats at various bit depths (8, 10, 12, 14, 16-bit including packed variants), and YUV/YCbCr formats (Yuv422, Yuv444, YCbCr601, YCbCr709 in various component orderings). The 'x' markers in the grid cells indicate which source-to-destination format conversions are supported by the VimbaX image transformation library.

**Key concepts:** `Pixel format transformations`, `Mono8, Mono10, Mono12, Mono14, Mono16 formats`, `Color8, Color10, Color12, Color14, Color16 formats`, `Bayer pattern formats (BayerXY*)`, `YUV/YCbCr color space formats (Yuv422, Yuv444, YCbCr601, YCbCr709)`, `Packed format variants`, `Source to destination format compatibility matrix`
