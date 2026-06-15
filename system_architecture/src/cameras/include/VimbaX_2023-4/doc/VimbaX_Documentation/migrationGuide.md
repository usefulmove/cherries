---
title: Vimba X Migration Guide from Vimba
description: Official documentation guide for migrating applications from Vimba SDK
  to Vimba X SDK, covering API changes, function mappings, and GenTL module architecture
  changes.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/migrationGuide.html
tags:
- vimba-x
- migration
- sdk-documentation
- api-reference
- camera-sdk
related: []
last_analyzed: '2026-03-09T07:58:33Z'
---

# Vimba X Migration Guide from Vimba

This HTML documentation provides a comprehensive migration guide for developers transitioning from the previous Vimba SDK to Vimba X (version 2023-4). The guide covers high-level architectural changes including the adoption of GenICam-compliant GenTL modules that separate features across TransportLayer, Interface, Camera, Stream, and LocalDevice modules. It documents API changes for C, C++, and Python including renamed structs, modified functions like VmbStartup() and VmbShutdown(), new query functions, and removed functionality. Key migration topics include chunk data replacing ancillary data, changes to frame acquisition workflows, event handling modifications, and the removal of direct register read/write functions. The guide emphasizes that Vimba X is not backward compatible and recommends using provided examples for migration.

**Key concepts:** `VmbC API replaces VimbaC`, `GenTL module architecture (TransportLayer, Interface, Camera, Stream, LocalDevice)`, `GenICam compliance changes`, `Chunk data replaces AncillaryData`, `Modified startup/shutdown functions with TL control`, `Frame acquisition changes (announce, revoke, capture)`, `Event handling changes and new discovery events`, `Removal of register read/write functions`, `C++ API requires C++11 or higher`, `Visual Studio 2017 or higher required`

## Sections

- **Migration from Vimba** — 
- **General tips** — 
- **High-level changes** — 
- **Using Vimba X functions** — 
- **Previous Vimba System functions** — 
- **Startup, Open, Close** — 
- **Query features and camera info** — 
- **Frame acquisition** — 
- **Chunk replaces AncillaryData** — 
- **Events** — 
- **Registers Read/Write** — 
