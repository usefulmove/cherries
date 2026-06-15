---
title: Copyright Linter Test
description: A pytest test that verifies all source files in the cherry_detection
  package contain valid copyright headers, using the ament_copyright linter.
source: docs/system_architecture/cherry_detection/test/test_copyright.md
tags:
- testing
- linting
- copyright
- ament
- ros2
related: []
last_analyzed: '2026-03-09T07:42:58Z'
---

# Copyright Linter Test

This document describes a pytest test function that runs the ament_copyright linter against the cherry_detection package's source and test directories to ensure all files contain proper copyright headers. The test is currently decorated with @pytest.mark.skip because the generated source files do not yet have copyright headers in place. It also includes @pytest.mark.copyright and @pytest.mark.linter markers for selective test execution, and asserts a zero return code from the linter to indicate no errors.

**Key concepts:** `ament_copyright linter`, `copyright header enforcement`, `pytest markers`, `skipped test`, `ROS2 package compliance`

## Sections

- **Copyright Linter Test** — Describes the test_copyright pytest function that validates copyright headers using the ament_copyright linter, noting it is currently skipped due to missing headers in generated files.
- **Exports** — Lists the test_copyright function as the exported test that runs the ament_copyright linter on package directories.
- **Dependencies** — Identifies ament_copyright and pytest as the required dependencies for this test.
