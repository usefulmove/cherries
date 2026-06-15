---
title: VimbaX Advanced Trigger Presets
description: Defines preset configurations for advanced camera triggering modes in
  VimbaX, including software triggers, hardware I/O line triggers, timers, and counters.
source: src/cameras/include/VimbaX_2023-4/bin/plugins/AdvTrigger_Presets.json
tags:
- vimbax
- camera-trigger
- presets
- industrial-camera
- allied-vision
related: []
last_analyzed: '2026-03-09T07:55:35Z'
---

# VimbaX Advanced Trigger Presets

This JSON configuration file defines preset trigger configurations for the VimbaX SDK's Advanced Trigger plugin. It provides ready-to-use presets for common camera triggering scenarios including software-based frame triggers, hardware I/O line edge and level triggers, bulk acquisition mode, PWM signal generation on output lines, and event counting. Each preset configures multiple camera features like trigger selectors, line modes, timers, and counters with appropriate values for specific use cases.

**Key concepts:** `TriggerSelector`, `TriggerSource`, `TriggerMode`, `LineSelector`, `TimerSelector`, `CounterSelector`, `Frame triggering`, `PWM output`, `Hardware I/O lines`

## Sections

- **presets** — Lists all available preset names that can be selected in the Advanced Trigger interface.
- **Turn Off All** — Disables all trigger sources, sets lines to input mode, and turns off timers and counters while preserving other trigger settings.
- **Reset All** — Completely resets all trigger settings to default values including delays, durations, debounce modes, and counter values.
- **Custom** — Placeholder preset that becomes active when users make manual trigger changes without modifying existing settings.
- **Frame Trigger Software** — Configures software-triggered frame acquisition with rising edge activation in continuous acquisition mode.
- **Frame Trigger IO Line0 Edge** — Configures hardware edge-triggered frame acquisition using Line0 input with rising edge activation.
- **Frame Trigger IO Line0 Level** — Configures level-triggered frame acquisition using Line0 where exposure duration matches the input signal width.
- **Frame Trigger IO Line0 Bulk** — Configures multi-frame bulk acquisition mode that captures 10 frames per trigger event on Line0.
- **PWM IO Line1 timer0 1khz 50%** — Generates a 1kHz PWM signal with 50% duty cycle on Line1 output using Timer0.
- **Counter IO Line0** — Configures Counter0 to count rising edge events on Line0 input for event counting applications.
