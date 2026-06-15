---
title: 'ROS Service Definition: Trigger'
description: A ROS 2 service definition file (.srv) for a simple Trigger service that
  accepts an integer ID as a request and returns an integer status as a response.
source: docs/system_architecture/cherry_interfaces/srv/Trigger.md
tags:
- ros2
- service
- robotics
- middleware
- interface
related: []
last_analyzed: '2026-03-09T07:43:51Z'
---

# ROS Service Definition: Trigger

This document describes the Trigger service definition file belonging to the cherry_interfaces ROS 2 package. The service defines a minimal interface with a request containing a single 32-bit integer field 'id' to identify the target or action to trigger, and a response containing a single 32-bit integer field 'status' to convey the outcome or state after execution. The service follows the standard ROS 2 .srv format with a three-dash separator between request and response sections, and would be compiled by rosidl_generate_interfaces into language-specific client/server stubs for use in ROS 2 nodes.

**Key concepts:** `ROS 2 .srv service definition format`, `Request field: int32 id — identifies the trigger target or action`, `Three-dash separator (---) dividing request from response`, `Response field: int32 status — returns the result/outcome of the trigger call`, `cherry_interfaces custom ROS 2 package`

## Sections

- **ROS Service Definition: Trigger** — Describes the Trigger service with its request (int32 id) and response (int32 status) fields, following ROS 2 .srv format conventions.
- **Dependencies** — Lists the build-time and runtime dependencies including rosidl_default_generators, rosidl_default_runtime, and the parent cherry_interfaces package.
