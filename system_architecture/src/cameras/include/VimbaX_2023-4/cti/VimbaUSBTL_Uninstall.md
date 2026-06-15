---
title: VimbaUSBTL Uninstall Script
description: Shell script for uninstalling the Vimba USB Transport Layer by removing
  GenICam GenTL environment variable exports and udev rules.
source: src/cameras/include/VimbaX_2023-4/cti/VimbaUSBTL_Uninstall.sh
tags:
- shell-script
- uninstaller
- vimba
- usb-camera
- genicam
related: []
last_analyzed: '2026-03-09T07:55:44Z'
---

# VimbaUSBTL Uninstall Script

This is an uninstallation shell script for the Allied Vision Vimba USB Transport Layer (AVTUSBTL). It removes the previously installed startup script that exports the GENICAM_GENTL64_PATH environment variable from /etc/profile.d/ and removes udev rules from /etc/udev/rules.d/. The script requires root privileges and supports x86_64/amd64 and aarch64 architectures. It's part of the VimbaX SDK for interfacing with Allied Vision USB cameras through the GenICam standard.

**Key concepts:** `GenICam GenTL transport layer`, `GENICAM_GENTL64_PATH environment variable`, `udev rules management`, `system profile scripts`, `root privilege requirement`
