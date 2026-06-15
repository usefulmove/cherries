---
title: VimbaX USB Transport Layer Installation Script
description: Shell script that configures system environment variables and udev rules
  for Allied Vision USB camera transport layer installation.
source: src/cameras/include/VimbaX_2023-4/cti/VimbaUSBTL_Install.sh
tags:
- shell-script
- camera-driver
- vimba-sdk
- usb-device
- linux-installation
related: []
last_analyzed: '2026-03-09T07:55:46Z'
---

# VimbaX USB Transport Layer Installation Script

This installation script sets up the VimbaX USB Transport Layer for Allied Vision cameras on Linux systems. It detects the system architecture (x86_64 or aarch64), requires root privileges, and performs two main configuration tasks: 1) Creates a profile script in /etc/profile.d/ to export the GENICAM_GENTL64_PATH environment variable needed for GenICam-compliant camera discovery, and 2) Creates udev rules in /etc/udev/rules.d/ to set proper USB device permissions (mode 0666) for Allied Vision USB cameras (vendor ID 1ab2). A system reboot is required after installation for the changes to take effect.

**Key concepts:** `GenICam GenTL path configuration`, `udev rules for USB device permissions`, `system architecture detection (x86/ARM)`, `profile.d environment variable export`, `Allied Vision USB camera setup`
