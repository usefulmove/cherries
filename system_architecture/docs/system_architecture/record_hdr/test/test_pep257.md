---
title: PEP 257 Docstring Style Linter Test
description: A pytest test that enforces PEP 257 docstring style compliance across
  the package source and test directories using the ament_pep257 linter.
source: docs/system_architecture/record_hdr/test/test_pep257.md
tags:
- testing
- linting
- pep257
- ament
- ros2
related: []
last_analyzed: '2026-03-09T07:45:21Z'
---

# PEP 257 Docstring Style Linter Test

This document describes a pytest test function `test_pep257` that runs the `ament_pep257` linter against the package's source and test directories. It asserts that the linter returns a zero exit code, indicating no docstring style violations. The test uses pytest markers (`@pytest.mark.linter` and `@pytest.mark.pep257`) to enable selective execution by category. This is a standard boilerplate linting test used in ROS 2 / ament-based Python packages to ensure PEP 257 compliance.

**Key concepts:** `PEP 257 docstring conventions`, `ament linting framework`, `pytest markers`, `code style enforcement`

## Sections

- **PEP 257 Docstring Style Linter Test** — Describes a pytest test that uses ament_pep257 to enforce PEP 257 docstring style compliance in ROS 2 packages.
- **Exports** — Lists the exported `test_pep257()` function that runs the linter and asserts compliance.
- **Dependencies** — Lists the external dependencies: `ament_pep257` and `pytest`.
