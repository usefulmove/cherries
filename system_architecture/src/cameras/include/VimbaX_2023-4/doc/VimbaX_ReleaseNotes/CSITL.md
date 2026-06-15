---
title: VimbaX CSI Transport Layer Release Notes
description: Release notes documentation for the Vimba CSI-2 Transport Layer (VimbaCSITL.cti)
  version 2.0.0, part of Allied Vision's VimbaX SDK for NVIDIA Jetson and embedded
  systems.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_ReleaseNotes/CSITL.html
tags:
- vimbax
- csi-camera
- nvidia-jetson
- release-notes
- allied-vision
related: []
last_analyzed: '2026-03-09T07:59:07Z'
---

# VimbaX CSI Transport Layer Release Notes

This HTML document contains the official release notes for the Vimba CSI Transport Layer version 2.0.0, a component of Allied Vision's VimbaX SDK (version 2023-4). The document covers supported hardware including NVIDIA Jetson AGX Orin Developer Kit with JetPack 5.1.0, installation prerequisites, known issues (such as dropped frames, switching between GenICam and V4L2 modes requiring reboot, and long exposure time issues), and a complete version history from 1.0.0 through 2.0.0. Version 2.0.0 added streaming support for i.MX 8M Plus EVK and Xilinx ZCU106 boards via their Yocto project. The documentation is built with Sphinx and uses Read the Docs theme with dark mode support.

**Key concepts:** `CSI-2 Transport Layer (VimbaCSITL.cti) version 2.0.0`, `NVIDIA Jetson AGX Orin support`, `JetPack 5.1.0 (L4T 35.1.0) compatibility`, `Allied Vision Alvium camera driver`, `i.MX 8M Plus EVK and Xilinx ZCU106 streaming support`, `GenICam for CSI-2 interface`, `V4L2 camera mode switching`

## Sections

- **Components and Version Reference** — Lists CSI-2 Transport Layer component version (2.0.0)
- **Supported hardware and driver** — Details tested platforms including NVIDIA Jetson AGX Orin with JetPack 5.1.0
- **Installation** — Prerequisites for installation including JetPack and Alvium camera driver
- **Correlations with other Allied Vision Software Packages** — Notes on coexistence with Vimba and Vimba X
- **Known issues** — Documents frame dropping, GenICam/V4L2 switching, and long exposure issues
- **Changes and release history** — Version history from 1.0.0 to 2.0.0 with feature changes and fixes
