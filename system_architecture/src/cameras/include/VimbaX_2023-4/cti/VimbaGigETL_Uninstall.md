---
title: VimbaGigETL Uninstall Script
description: Shell script for uninstalling the Vimba GigE Transport Layer by removing
  the startup script that exports GenICam GenTL environment variables.
source: src/cameras/include/VimbaX_2023-4/cti/VimbaGigETL_Uninstall.sh
tags:
- shell-script
- uninstaller
- vimba
- genicam
- gentl
related: []
last_analyzed: '2026-03-09T07:55:40Z'
---

# VimbaGigETL Uninstall Script

This is a shell uninstallation script for the Allied Vision VimbaGigETL (GigE Transport Layer) component. It detects the system architecture (x86_64 or aarch64), verifies root privileges, and removes the 64-bit GenTL registration script from /etc/profile.d/. The script is part of the VimbaX 2023-4 camera SDK and manages the cleanup of GENICAM_GENTL64_PATH environment variable configuration that was set up during installation.

**Key concepts:** `GenICam GenTL path configuration`, `GigE Transport Layer`, `System profile.d scripts`, `Architecture detection (x86/arm)`, `Root privilege verification`
