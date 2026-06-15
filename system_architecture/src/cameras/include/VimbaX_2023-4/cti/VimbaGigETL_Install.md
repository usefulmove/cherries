---
title: VimbaGigE Transport Layer Installation Script
description: Shell script for setting up the Vimba GigE Transport Layer by creating
  startup scripts that export GenICam environment variables for the AVT GigE transport
  layer.
source: src/cameras/include/VimbaX_2023-4/cti/VimbaGigETL_Install.sh
tags:
- shell-script
- installation
- genicam
- gige-camera
- vimba
related: []
last_analyzed: '2026-03-09T07:55:40Z'
---

# VimbaGigE Transport Layer Installation Script

This is an installation shell script for the Allied Vision Technologies Vimba GigE Transport Layer. It detects the system architecture (supporting x86_64/amd64 and aarch64), verifies root privileges are being used, and creates a startup script in /etc/profile.d/ that exports the GENICAM_GENTL64_PATH environment variable. This environment variable is required by GenICam-compliant applications to locate the GigE transport layer library. The script registers the current working directory as the transport layer path and requires the user to log off for changes to take effect.

**Key concepts:** `GenICam transport layer registration`, `GENICAM_GENTL64_PATH environment variable`, `profile.d startup script creation`, `architecture detection (x86_64, aarch64)`, `root privilege verification`
