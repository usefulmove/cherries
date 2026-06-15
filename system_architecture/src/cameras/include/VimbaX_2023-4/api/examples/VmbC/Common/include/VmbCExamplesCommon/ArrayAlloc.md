---
title: VmbC Array Allocation Helper
description: A header file providing a convenience macro for allocating typed arrays
  using malloc in VimbaX camera SDK examples.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/Common/include/VmbCExamplesCommon/ArrayAlloc.h
tags:
- c-header
- memory-allocation
- vimbax-sdk
- camera-api
- utility-macro
related: []
last_analyzed: '2026-03-09T07:48:22Z'
---

# VmbC Array Allocation Helper

This header file is part of the Allied Vision VimbaX camera SDK examples. It provides a simple utility macro VMB_MALLOC_ARRAY that wraps the standard malloc function to allocate memory for typed arrays. The macro takes a type and size parameter, automatically calculating the correct byte size and casting the returned pointer to the appropriate type. It also includes a fallback NULL definition for compatibility.

**Key concepts:** `typed array allocation`, `malloc wrapper macro`, `C memory management`, `VimbaX SDK utilities`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `#define VMB_MALLOC_ARRAY(type, size) ((type*) malloc(size * sizeof(type)))` | constant | Macro that allocates memory for an array of the specified size and type using malloc, returning a properly typed pointer |
| `#define NULL 0` | constant | Fallback definition of NULL as 0, only defined if not already present |
