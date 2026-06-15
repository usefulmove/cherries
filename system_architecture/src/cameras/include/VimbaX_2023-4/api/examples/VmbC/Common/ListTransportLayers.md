---
title: ListTransportLayers
description: Utility function to enumerate and list available VimbaX transport layers
  for camera discovery.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/ListTransportLayers.c
tags:
- vimbax
- camera-sdk
- transport-layer
- c-api
- enumeration
related: []
last_analyzed: '2026-03-09T07:48:13Z'
---

# ListTransportLayers

This C source file provides a utility function `ListTransportLayers` for the VimbaX SDK examples. It implements a common pattern for enumerating transport layers by first querying the count of available transport layers, allocating an appropriately sized buffer using `VMB_MALLOC_ARRAY`, and then populating it with transport layer information via `VmbTransportLayersList`. The function handles various error conditions including no transport layers found, insufficient memory, and partial data retrieval scenarios.

**Key concepts:** `VimbaX transport layer enumeration`, `VmbTransportLayersList API usage`, `dynamic memory allocation for transport layer info`, `error handling for VmbError_t`, `two-pass query pattern (count then populate)`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `VmbError_t ListTransportLayers(VmbTransportLayerInfo_t** transportLayers, VmbUint32_t* count)` | function | Retrieves a list of available transport layers. Allocates memory for the transport layer info array and returns it via the transportLayers output parameter. The caller is responsible for freeing the allocated memory. Returns VmbErrorSuccess on success, VmbErrorNotFound if no transport layers exist, or VmbErrorResources if memory allocation fails. |

## Dependencies

`VmbC/VmbC.h`
