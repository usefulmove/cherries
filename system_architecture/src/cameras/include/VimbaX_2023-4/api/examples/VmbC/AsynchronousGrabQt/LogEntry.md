---
title: LogEntry Class for VimbaX Event Logging
description: Defines a LogEntry class used to store log messages with associated VimbaX
  error codes for the event log view in the AsynchronousGrabQt example.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbC/AsynchronousGrabQt/LogEntry.h
tags:
- vimbax
- camera-sdk
- logging
- qt-example
- cpp-header
related: []
last_analyzed: '2026-03-09T07:47:19Z'
---

# LogEntry Class for VimbaX Event Logging

This header file defines the LogEntry class within the VmbC::Examples namespace, part of the VimbaX SDK's AsynchronousGrabQt example application. The class encapsulates a log message (stored as a string) along with an optional VimbaX error code (VmbError_t), defaulting to VmbErrorSuccess. It provides getter methods GetMessage() and GetErrorCode() to retrieve the stored message and error code respectively. This class is designed to be used with an event log view component for displaying status messages and errors during asynchronous camera frame grabbing operations.

**Key concepts:** `log entry encapsulation`, `VimbaX error codes`, `message storage with error association`, `event logging`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class LogEntry` | class | Encapsulates a log message string with an associated VimbaX error code for event logging purposes |
| `LogEntry(std::string const& message, VmbError_t error = VmbErrorSuccess)` | method | Constructor that initializes the log entry with a message and optional error code (defaults to VmbErrorSuccess) |
| `std::string const& GetMessage() const noexcept` | method | Returns a const reference to the stored log message |
| `VmbError_t GetErrorCode() const noexcept` | method | Returns the VimbaX error code associated with this log entry |

## Dependencies

`VmbC/VmbC.h`
