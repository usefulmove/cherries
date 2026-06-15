---
title: Cherry_cpp Constructor Implementation
description: Implements the constructor for the Cherry_cpp class, initializing its
  X, Y coordinate and Type fields via member initializer list.
source: docs/system_architecture/tracking_projector/src/cherry_cpp.md
tags:
- cpp
- tracking
- data-model
- constructor
related: []
last_analyzed: '2026-03-09T07:45:57Z'
---

# Cherry_cpp Constructor Implementation

This documentation describes the Cherry_cpp constructor implementation used in a tracking projector system. The Cherry_cpp class represents a tracked point or feature with 2D position coordinates (X, Y) and an integer type discriminator. The constructor uses a member initializer list to set up these fields, with an empty constructor body. The class is designed for representing discrete trackable entities such as keypoints, blobs, or markers that are characterized by spatial coordinates and category type.

**Key concepts:** `Cherry_cpp class instantiation`, `Member initializer list`, `2D spatial coordinates`, `Object type classification`

## Sections

- **Cherry_cpp Constructor Implementation** — Describes the Cherry_cpp constructor that initializes a tracked object with 2D position and type discriminator using member initializer list.
- **Exports** — Documents the Cherry_cpp constructor method signature taking x, y coordinates and type parameter.
- **Dependencies** — Lists the dependency on the cherry_cpp.hpp header file.
