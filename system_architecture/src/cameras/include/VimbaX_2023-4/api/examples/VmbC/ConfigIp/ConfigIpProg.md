---
title: ConfigIp Example Program Header
description: Header file declaring the ConfigIpProg function for configuring IP settings
  on Allied Vision cameras using the VmbC API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ConfigIp/ConfigIpProg.h
tags:
- vimbax
- camera-configuration
- ip-configuration
- c-header
- allied-vision
related: []
last_analyzed: '2026-03-09T07:48:46Z'
---

# ConfigIp Example Program Header

This header file is part of the VimbaX SDK examples and declares the ConfigIpProg function, which provides functionality to configure IP settings for Allied Vision cameras. The function supports three IP configuration modes: setting a persistent static IP address with subnet mask and optional gateway, enabling DHCP by passing 'dhcp' as the IP parameter, or configuring Link-Local Address (LLA) by passing NULL. The configuration is written to the camera's IP configuration registers and persists across power cycles.

**Key concepts:** `camera IP configuration`, `persistent IP settings`, `DHCP configuration`, `Link-Local Address (LLA)`, `VmbC API usage`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int ConfigIpProg(const char* const cameraId, const char* const ip, const char* const subnet, const char* const gateway)` | function | Sets an IP configuration for a camera identified by its ID. Starts the VmbC API, writes IP configuration to the camera's registers, and the configuration persists after power-cycle. Supports persistent IP, DHCP (ip='dhcp'), or LLA (ip=NULL) modes. |
