---
title: VmbC ChunkAccess Example Main Entry Point
description: Main entry point for the VimbaX ChunkAccess example demonstrating chunk
  data access with the Vmb C API.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/ChunkAccess/main.c
tags:
- vimbax
- camera-api
- chunk-access
- c-example
- embedded-vision
related: []
last_analyzed: '2026-03-09T07:47:59Z'
---

# VmbC ChunkAccess Example Main Entry Point

This file serves as the main entry point for the VimbaX ChunkAccess example application. It is part of the Allied Vision VimbaX SDK examples demonstrating how to access chunk data from camera images using the Vmb C API. The main function prints a header banner and delegates execution to the ChunkAccessProg() function defined in ChunkAccessProg.h. The program does not expect any command-line arguments and will notify the user if any are provided.

**Key concepts:** `VimbaX SDK`, `chunk data access`, `camera programming`, `Allied Vision cameras`, `C API example`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `int main(int argc, char* argv[])` | function | Main entry point that initializes the chunk access example, prints a banner, and calls ChunkAccessProg() to run the demonstration. |
