---
title: CherryBuffer Unit Tests
description: Unit tests for the CherryBuffer ROS 2 node, validating buffer add/pop
  operations with encoder-count-based position adjustment and service callback behaviour.
source: docs/system_architecture/cherry_buffer/cherry_buffer/test_cherry_buffer.md
tags:
- ros2
- unit-testing
- robotics
- sensor-data
- cherry-detection
related: []
last_analyzed: '2026-03-09T07:42:31Z'
---

# CherryBuffer Unit Tests

This documentation describes the unit test suite for the CherryBuffer module, a ROS 2 node used in cherry-harvesting or cherry-sorting robotic systems. The tests verify that CherryArray messages containing multiple cherry detections can be stored in a buffer and retrieved with encoder-count-based positional adjustments. Test cases validate buffer push/pop operations at matching and differing encoder counts, as well as direct testing of detection_callback and get_callback service handlers. The test setup initializes nine cherry objects with varying coordinates and classification types, while teardown cleanly destroys the ROS 2 node.

**Key concepts:** `CherryBuffer ROS 2 node under test`, `Cherry detection message types (CherryArray, Cherry)`, `Encoder-count-based positional offset calculation`, `Buffer push/pop with reference encoder position`, `ROS 2 service request/response testing (GetCherryBuffer)`, `Cherry classification constants (CHERRY_CLEAN, CHERRY_PIT, CHERRY_SIDE, CHERRY_MAYBE, CHERRY_STEM)`

## Sections

- **CherryBuffer Unit Tests** — Overview of the unittest suite for the CherryBuffer ROS 2 node, covering buffer operations and encoder-based position adjustments.
- **Exports** — Documents the test case class, setUp method, two test methods (test_add_to_buffer, test_services), teardown destructor, and main entry point function.
- **Dependencies** — Lists required packages: rclpy, cherry_buffer, sensor_msgs, builtin_interfaces, and cherry_interfaces.
