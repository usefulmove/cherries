---
title: VimbaX Bandwidth Manager Plugin Settings
description: Configuration file that defines feature and category filters for the
  VimbaX Bandwidth Manager plugin, controlling which camera features are exposed for
  bandwidth management.
source: src/cameras/include/VimbaX_2023-4/bin/plugins/BandwidthManager_Settings.json
tags:
- vimbax
- camera
- bandwidth-management
- gige-vision
- streaming
related: []
last_analyzed: '2026-03-09T07:55:31Z'
---

# VimbaX Bandwidth Manager Plugin Settings

This JSON configuration file defines the settings for the VimbaX Bandwidth Manager plugin. It specifies two filter lists: FeaturesFilter contains camera and stream features related to bandwidth control such as throughput limits, exposure settings, frame rates, packet sizes, and stream statistics. CategoriesFilter defines which feature categories from the camera and stream interfaces should be accessible, including stream settings, statistics, and image format controls. These filters determine what parameters the Bandwidth Manager plugin can monitor and adjust for optimizing data transfer from GigE Vision and USB cameras.

**Key concepts:** `FeaturesFilter`, `CategoriesFilter`, `DeviceLinkThroughputLimit`, `AcquisitionFrameRate`, `StreamBytesPerSecond`, `GevSCPSPacketSize`

## Sections

- **FeaturesFilter** — List of individual camera and stream features (such as throughput limits, exposure, frame rates, packet sizes, buffer counts, and statistics) that the Bandwidth Manager plugin can access and manage.
- **CategoriesFilter** — List of feature category paths from the Stream and Camera interfaces that define which groups of settings are available to the plugin for bandwidth management.
