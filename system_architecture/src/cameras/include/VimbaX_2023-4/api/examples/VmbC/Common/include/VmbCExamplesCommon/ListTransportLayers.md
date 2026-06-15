---
title: ListTransportLayers Header
description: Header file declaring a function to list available transport layers in
  the VimbaX camera SDK.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/include/VmbCExamplesCommon/ListTransportLayers.h
tags:
- vimbax
- camera-sdk
- transport-layer
- c-header
- allied-vision
related: []
last_analyzed: '2026-03-09T07:48:32Z'
---

# ListTransportLayers Header

This is a C header file for the VimbaX camera SDK examples. It declares the ListTransportLayers function which retrieves an array of available transport layers (camera interfaces) from the Vimba system. The function allocates memory using malloc and returns transport layer information via output parameters, following the VmbC API conventions. Transport layers represent different physical interfaces through which cameras can be connected (e.g., GigE, USB3).

**Key concepts:** `transport layer enumeration`, `VimbaX API`, `camera interface discovery`, `VmbC bindings`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbError_t ListTransportLayers(VmbTransportLayerInfo_t** transportLayers, VmbUint32_t* count)` | function | Gets a list of available transport layers, returning an array allocated via malloc and the count of layers found. Returns VmbErrorNotFound if no transport layers are available. |

## Dependencies

`VmbC`
