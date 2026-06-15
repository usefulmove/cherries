---
title: History – Tracking Projector History Header
description: Declares the History class, which records a short fixed-size history
  of integer counts and provides a simple prediction of future values based on past
  data.
source: docs/system_architecture/tracking_projector/include/tracking_projector/history.md
tags:
- tracking
- prediction
- history
- cpp-header
related: []
last_analyzed: '2026-03-09T07:45:50Z'
---

# History – Tracking Projector History Header

This documentation describes the C++ header file for the History class within the tracking_projector package. The History class maintains a fixed-size array of 10 integer values representing recent observed counts, along with a 'last' field. It exposes two public methods: add() for appending new count values to the history buffer, and predict() for estimating future values a given number of steps ahead based on the stored history. The class is designed to smooth or extrapolate target-tracking data for projection purposes.

**Key concepts:** `fixed-size circular/rolling history buffer`, `integer count tracking`, `step-ahead prediction`, `header guard`, `History class`

## Sections

- **History – Tracking Projector History Header** — Introduces the History class and its purpose for maintaining a rolling history buffer and providing prediction capabilities.
- **Exports** — Lists the exported symbols including the History class, its constructor, and its add() and predict() methods.
