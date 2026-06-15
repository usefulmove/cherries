---
title: Frame – Cherry Classification Rendering Frame
description: Declares the Frame class, which encapsulates a single frame of cherry
  classification data and renders each cherry category (pit, clean, maybe, side) as
  separate QImage layers using configurable colors and a shared circle size.
source: docs/system_architecture/tracking_projector/include/tracking_projector/frame.md
tags:
- qt
- image-rendering
- cherry-sorting
- computer-vision
- ros
related: []
last_analyzed: '2026-03-09T07:45:50Z'
---

# Frame – Cherry Classification Rendering Frame

This document describes the Frame class in the tracking_projector package, which represents a single processing frame of cherry sorting data. Each Frame holds a collection of Cherry_cpp objects (classified cherries) and an encoder count for positional synchronization. The class maintains four separate QImage buffers—one each for pit, clean, maybe, and side classifications—painted with configurable QColor values via a private drawAll() method. Public accessors expose each image layer and the raw cherry/encoder data. A static circle_size member controls the rendered circle radius uniformly across all frame instances, and the class is designed to decouple rendering of classified cherries from downstream display or projection logic.

**Key concepts:** `Per-frame cherry classification rendering`, `Multiple QImage layers per category (pits, cleans, maybes, sides)`, `Encoder count for positional/synchronization reference`, `Configurable per-category QColor`, `Static shared circle size across all frames`, `Cherry_cpp domain object aggregation`

## Sections

- **Frame – Cherry Classification Rendering Frame** — Introduces the Frame class and its role in encapsulating cherry classification data with per-category QImage rendering layers.
- **Exports** — Lists the Frame class, its constructors, accessor methods for image layers and data, and static circle size getter/setter methods.
- **Dependencies** — Documents the external dependencies including Qt5 (QImage, QPainter, QColor) and cherry_cpp.hpp.
