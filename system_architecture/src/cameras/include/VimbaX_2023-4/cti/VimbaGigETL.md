---
title: VimbaX GigE Transport Layer Configuration
description: XML configuration file for the Vimba GigE Transport Layer (GenTL) that
  controls logging, network packet sizes, and device discovery settings for GigE Vision
  cameras.
source: src/cameras/include/VimbaX_2023-4/cti/VimbaGigETL.xml
tags:
- VimbaX
- GigE-Vision
- camera-configuration
- GenTL
- network-settings
related: []
last_analyzed: '2026-03-09T07:55:43Z'
---

# VimbaX GigE Transport Layer Configuration

This XML configuration file provides settings for the VimbaX GigE Transport Layer (CTI - Common Transport Interface). It allows configuration of logging options (filename and append mode), network packet size (MTU/GVSP packet size ranging from 500 to 16384 bytes with Jumbo Frame support), and device discovery modes for GigE Vision cameras. The default configuration has logging disabled, uses default packet size of 8228 bytes (requiring Jumbo Frames), and sets device discovery to 'Auto' mode for permanent camera detection. Most settings are commented out, relying on defaults, with only DefaultDeviceDiscovery actively set to 'Auto'.

**Key concepts:** `GigE camera discovery modes (Off, Once, Auto)`, `Ethernet payload/MTU configuration (Jumbo Frames)`, `Transport layer logging`, `GVSPPacketSize settings`, `GenTL interface configuration`

## Dependencies

`VimbaX SDK`, `GigE Vision network infrastructure`

## Sections

- **LogFileName** — Configuration for enabling and naming the transport layer log file
- **AppendLog** — Configuration for log file append vs reset behavior
- **DefaultPacketSize** — Ethernet frame payload size / MTU configuration (500-16384 bytes)
- **DefaultDeviceDiscovery** — GigE camera detection mode settings (Off/Once/Auto)
