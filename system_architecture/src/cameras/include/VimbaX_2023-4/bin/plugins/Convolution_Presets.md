---
title: VimbaX Convolution Filter Presets
description: Defines preset convolution kernel matrices for image processing filters
  used by the VimbaX camera plugin system.
source: src/cameras/include/VimbaX_2023-4/bin/plugins/Convolution_Presets.json
tags:
- vimbax
- image-processing
- convolution
- camera
- filters
related: []
last_analyzed: '2026-03-09T07:55:31Z'
---

# VimbaX Convolution Filter Presets

This JSON configuration file defines preset convolution filter kernels for the VimbaX 2023-4 camera SDK plugin system. It includes eight standard image processing filters: Identity (passthrough), Ridge detection, Sharpen, Box blur, Gaussian blur (in 3x3 and 5x5 variants), and Sobel edge detection filters (horizontal and vertical). Each preset is represented as a flattened 5x5 kernel matrix (25 values) that can be applied to camera image data for real-time processing effects.

**Key concepts:** `5x5 convolution kernels`, `Identity filter`, `Ridge detection`, `Sharpen filter`, `Box blur`, `Gaussian blur (3x3 and 5x5)`, `Sobel edge detection (horizontal/vertical)`

## Sections

- **presets** — Lists the available convolution filter preset names that can be selected.
- **Identity** — A passthrough kernel that preserves the original image unchanged.
- **Sharpen** — A sharpening filter kernel that enhances edges and details.
- **Ridge detection** — A kernel for detecting ridges and edges in all directions.
- **Box blur** — A simple averaging blur filter using uniform 3x3 weights.
- **Gaussian blur 3x3** — A Gaussian-weighted blur kernel approximating a 3x3 Gaussian distribution.
- **Gaussian blur 5x5** — A full 5x5 Gaussian-weighted blur kernel for stronger smoothing.
- **Sobel horizontal** — A Sobel operator kernel for detecting horizontal edges.
- **Sobel vertical** — A Sobel operator kernel for detecting vertical edges.
