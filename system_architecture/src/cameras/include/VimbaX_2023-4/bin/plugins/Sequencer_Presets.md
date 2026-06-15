---
title: VimbaX Camera Sequencer Presets
description: Configuration preset file for VimbaX camera sequencer defining sequencer
  sets with gain values and transition paths to safely control image acquisition sequences.
source: src/cameras/include/VimbaX_2023-4/bin/plugins/Sequencer_Presets.json
tags:
- vimbax
- camera
- sequencer
- image-acquisition
- allied-vision
related: []
last_analyzed: '2026-03-09T07:55:36Z'
---

# VimbaX Camera Sequencer Presets

This JSON configuration file defines sequencer presets for VimbaX 2023-4 camera SDK. It configures a sequence of 4 sets (Set0-Set3) with different Gain values (3.0, 0.0, 6.0, 0.0) that cycle during image acquisition. Each set defines paths with trigger sources (ExposureActive and SoftwareSignal1) that determine transitions between sets. Set3 serves as a special 'parking set' to safely stop image acquisition without crashes. The configuration demonstrates a pattern to prevent crashes that can occur when stopping acquisition during an active exposure, using a software-controllable signal to transition to the parking set before executing the AcquisitionStop command.

**Key concepts:** `SequencerSetStart`, `SequencerSetSelector`, `Gain feature configuration`, `SequencerPathSelector`, `SequencerTriggerSource`, `Parking set for safe acquisition stop`

## Sections

- **SequencerSetStart** — Defines the starting sequencer set index (Set0) for the acquisition sequence.
- **Set0 (SequencerSetSelector: 0)** — First sequencer set with Gain 3.0, configured to transition to Set1 on exposure falling edge or to parking Set3 via SoftwareSignal1.
- **Set1 (SequencerSetSelector: 1)** — Second sequencer set with Gain 0.0, configured to transition to Set2 on exposure falling edge or to parking Set3 via SoftwareSignal1.
- **Set2 (SequencerSetSelector: 2)** — Third sequencer set with Gain 6.0, configured to loop back to Set0 on exposure falling edge or to parking Set3 via SoftwareSignal1.
- **Set3 - Parking Set (SequencerSetSelector: 3)** — Parking set with Gain 0.0 and no transition paths, used as a safe state to stop image acquisition without causing crashes.
