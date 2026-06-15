---
title: ForceIp Example Program Header
description: Header file declaring the ForceIpProg function for modifying IP configuration
  of cameras via the VimbaX SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ForceIp/ForceIpProg.h
tags:
- vimbax
- camera-api
- network-configuration
- force-ip
- header
related: []
last_analyzed: '2026-03-09T07:49:04Z'
---

# ForceIp Example Program Header

This header file is part of the VimbaX SDK examples and declares the ForceIpProg function, which sends a Force IP command to modify the network configuration (IP address, subnet mask, and gateway) of a camera identified by its MAC address. The function attempts the operation through connected interfaces first, then falls back to trying all Vimba X and Vimba GigE transport layers. The configuration changes are temporary and will be lost after a power-cycle of the camera.

**Key concepts:** `Force IP command`, `Camera network configuration`, `MAC address identification`, `IP address assignment`, `GigE transport layer`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int ForceIpProg(const char* const strMAC, const char* const strIP, const char* const strSubnet, const char* const strGateway)` | function | Modifies the IP configuration for a camera identified by the given MAC address by sending a Force IP command. Parameters include MAC address, IP address, subnet mask, and optional gateway. Returns a code suitable for main() return value. |
