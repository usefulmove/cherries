---
title: Qt UI Definition for VimbaX Asynchronous Grab Example
description: A Qt Designer UI file defining the main window layout for the VimbaX
  SDK's asynchronous camera grab example application.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/AsynchronousGrabQt/UI/res/AsynchronousGrabGui.ui
tags:
- qt
- ui-design
- vimbax-sdk
- camera-interface
- xml
related: []
last_analyzed: '2026-03-09T07:47:41Z'
---

# Qt UI Definition for VimbaX Asynchronous Grab Example

This is a Qt Designer UI file (XML format) that defines the graphical user interface for the VimbaX SDK's AsynchronousGrabQt example application. The interface consists of a QMainWindow (1040x780 pixels) with a grid layout containing four main components: a camera selection tree view (QTreeView) on the left side, a custom ImageLabel widget for displaying camera frames in the center-right area, a 'Start Acquisition' button below the camera list, and an event log table (QTableView) at the bottom right. The layout uses a 2x2 grid with configurable stretching and minimum size constraints to provide a functional camera control and monitoring interface.

**Key concepts:** `QMainWindow layout`, `QTreeView for camera selection`, `ImageLabel for rendering camera output`, `QPushButton for acquisition control`, `QTableView for event logging`, `QGridLayout structure`

## Dependencies

`Qt framework (version 4.0+)`, `ImageLabel custom widget class`

## Sections

- **Main Window Definition** — XML header and QMainWindow properties including geometry, size policy, and window title
- **Image Render Label** — Custom ImageLabel widget for displaying camera frames
- **Acquisition Button** — Start/Stop acquisition button control
- **Camera Selection Tree** — QTreeView for selecting cameras from available devices
- **Event Log Table** — QTableView for displaying acquisition events and status messages
