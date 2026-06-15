---
title: PEP 257 Docstring Style Linter Test
description: A pytest test that enforces PEP 257 docstring conventions across the
  fanuc_comms package using the ament_pep257 linter.
source: docs/system_architecture/fanuc_comms/test/test_pep257.md
tags:
- linting
- pep257
- testing
- ros2
- code-style
related: []
last_analyzed: '2026-03-09T07:44:19Z'
---

# PEP 257 Docstring Style Linter Test

This documentation describes a pytest test function that runs the ament_pep257 linter against the fanuc_comms package source and test directories to verify compliance with PEP 257 docstring style conventions. It is a standard ROS 2 (ament-based) linter test that invokes ament_pep257.main with the current and test directories as targets, asserting a zero return code to indicate no docstring style violations were found. The test is tagged with the linter and pep257 pytest markers, which allows it to be selectively included or excluded during test runs.

**Key concepts:** `PEP 257 docstring conventions`, `ament linter integration`, `pytest markers`, `code style enforcement`

## Sections

- **PEP 257 Docstring Style Linter Test** — Describes the purpose and functionality of the test that runs the ament_pep257 linter to check for PEP 257 compliance.
- **Exports** — Documents the test_pep257() function that runs the ament_pep257 linter on source directories and asserts a return code of 0.
- **Dependencies** — Lists the required dependencies: ament_pep257 and pytest.
