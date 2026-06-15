---
title: TransportLayerTypeToString
description: Header file declaring a utility function to convert VimbaX transport
  layer type enum values to human-readable strings.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/include/VmbCExamplesCommon/TransportLayerTypeToString.h
tags:
- vimbax
- camera-api
- transport-layer
- utility
- c-header
related: []
last_analyzed: '2026-03-09T07:48:31Z'
---

# TransportLayerTypeToString

This header file is part of the VimbaX 2023-4 SDK examples common utilities. It declares a single function `TransportLayerTypeToString` that converts `VmbTransportLayerType_t` enum constants (representing different camera transport layer types like GigE, USB3, etc.) into human-readable string representations. The file includes the VmbCTypeDefinitions.h header for the VmbTransportLayerType_t type definition.

**Key concepts:** `transport layer type conversion`, `VmbTransportLayerType_t enum`, `string representation`, `VimbaX SDK`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `char const* TransportLayerTypeToString(VmbTransportLayerType_t tlType)` | function | Converts a transport layer type enum constant to a human-readable string representation |

## Dependencies

`VmbC`
