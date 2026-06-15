---
title: VmbCPP Clock Utility Class
description: Platform-independent clock and sleep utility class for the VmbCPP library,
  providing time measurement and thread sleep functionality.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/Clock.h
tags:
- vimba-sdk
- clock
- timing
- platform-independent
- cpp
related: []
last_analyzed: '2026-03-09T07:53:47Z'
---

# VmbCPP Clock Utility Class

This header file defines the Clock class for the VmbCPP (Vimba C++) SDK from Allied Vision Technologies. The class provides platform-independent timing functionality including methods to measure elapsed time (Reset, SetStartTime, GetTime), get absolute time (GetAbsTime), and pause execution for specified durations (Sleep, SleepMS, SleepAbs). The class is marked as final to prevent inheritance and maintains internal state through a protected m_dStartTime member. It is intended for internal use within the VmbCPP implementation.

**Key concepts:** `platform-independent sleep`, `time measurement`, `absolute time tracking`, `start time management`, `millisecond sleep`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class Clock final` | class | Platform-independent clock utility class for time measurement and sleep operations |
| `Clock()` | method | Default constructor |
| `~Clock()` | method | Destructor |
| `void Reset()` | method | Resets the clock state |
| `void SetStartTime()` | method | Sets the start time to the current time |
| `void SetStartTime(double dStartTime)` | method | Sets the start time to a specific value |
| `double GetTime() const` | method | Returns elapsed time since start time was set |
| `static double GetAbsTime()` | method | Returns the current absolute time |
| `static void Sleep(double dTime)` | method | Pauses execution for the specified duration in seconds |
| `static void SleepMS(unsigned long nTimeMS)` | method | Pauses execution for the specified duration in milliseconds |
| `static void SleepAbs(double dAbsTime)` | method | Pauses execution until the specified absolute time is reached |
