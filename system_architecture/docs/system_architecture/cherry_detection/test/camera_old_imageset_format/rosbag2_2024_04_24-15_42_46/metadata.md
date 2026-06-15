---
title: ROS 2 Bag Metadata (cherry_detection old imageset format)
description: ROS 2 rosbag2 metadata file describing a recorded bag containing 210
  HDR image-set messages published on the /image_set topic, defining the bag's storage
  format, timing, topic configuration, and QoS profiles.
source: docs/system_architecture/cherry_detection/test/camera_old_imageset_format/rosbag2_2024_04_24-15_42_46/metadata.md
tags:
- ros2
- rosbag2
- metadata
- camera
- cherry-detection
related: []
last_analyzed: '2026-03-09T07:42:55Z'
---

# ROS 2 Bag Metadata (cherry_detection old imageset format)

This document describes the mandatory rosbag2 metadata descriptor for a ROS 2 bag recorded on 2024-04-24 as part of the cherry_detection test suite. The bag spans approximately 118.9 seconds and contains 210 messages on the /image_set topic using the custom cherry_interfaces/msg/ImageSetHdr message type for HDR multi-camera images. The metadata specifies SQLite3 storage with CDR encoding, no compression, and sensor-data QoS defaults (keep-last history, reliable reliability, volatile durability). This file is used by rosbag2 tooling for replay and inspection, and validates that the old ImageSet message format can still be processed correctly.

**Key concepts:** `rosbag2_bagfile_information version 5`, `sqlite3 storage backend`, `cherry_interfaces/msg/ImageSetHdr message type`, `CDR serialization format`, `QoS profile configuration`, `rosbag2 metadata schema`, `HDR multi-camera image set`

## Sections

- **ROS 2 Bag Metadata (cherry_detection old imageset format)** — Overview of the rosbag2 metadata file for a test dataset containing 210 HDR image-set messages from the cherry_detection package.
- **Sections** — Detailed breakdown of metadata components including bag descriptor, duration, starting time, topics with message counts, compression settings, and file paths.
