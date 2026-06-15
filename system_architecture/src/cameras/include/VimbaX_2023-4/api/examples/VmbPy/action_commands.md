---
title: VmbPy Action Commands Example
description: Example script demonstrating how to use VimbaX/VmbPy action commands
  to trigger camera frame acquisition on Allied Vision GigE cameras.
source: src/cameras/include/VimbaX_2023-4/api/examples/VmbPy/action_commands.py
tags:
- vimba
- camera-control
- action-commands
- gige-vision
- example
related: []
last_analyzed: '2026-03-09T07:50:15Z'
---

# VmbPy Action Commands Example

This example script demonstrates how to use Allied Vision's VmbPy (Vimba Python SDK) to send action commands to GigE Vision cameras. It configures a camera for action command triggering by setting up the trigger selector, source, mode, and action keys. The script enters a streaming mode where the user can interactively press 'a' to send action commands that trigger frame acquisition, or 'q' to quit. It includes GigE-specific packet size adjustment and handles frame reception through a callback handler.

**Key concepts:** `Action Commands`, `GigE Vision camera triggering`, `Frame streaming`, `Camera trigger configuration`, `VmbPy SDK usage`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `def print_preamble()` | function | Prints the example program banner/header to stdout. |
| `def print_usage()` | function | Prints command-line usage instructions for the script. |
| `def abort(reason: str, return_code: int = 1, usage: bool = False)` | function | Prints an error message and optionally usage instructions, then exits with the given return code. |
| `def parse_args() -> Optional[str]` | function | Parses command-line arguments, returning the camera ID if provided or None if not. |
| `def get_input() -> str` | function | Prompts the user and returns their input character ('a' to send action, 'q' to quit). |
| `def get_camera(camera_id: Optional[str]) -> Camera` | function | Returns a Camera object by ID, or the first available camera if no ID is provided. |
| `def frame_handler(cam: Camera, stream: Stream, frame: Frame)` | function | Callback function that prints frame ID when a complete frame is received and re-queues the frame. |
| `def main()` | function | Main entry point that sets up camera triggering, starts streaming, and processes user input for action commands. |

## Dependencies

`vmbpy`
