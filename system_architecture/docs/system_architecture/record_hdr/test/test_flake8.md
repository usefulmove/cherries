---
title: Flake8 Linter Test
description: A pytest-based test that runs flake8 code style checks on the package
  source using ament_flake8, asserting zero style errors or warnings.
source: docs/system_architecture/record_hdr/test/test_flake8.md
tags:
- linting
- flake8
- testing
- code-style
- ament
related: []
last_analyzed: '2026-03-09T07:45:23Z'
---

# Flake8 Linter Test

This documentation describes a pytest test file that performs flake8 code style linting on a ROS 2 / ament-based Python package. The test function `test_flake8` invokes `ament_flake8`'s `main_with_errors` to run a complete style check pass and asserts that no violations are detected. The test is decorated with `@pytest.mark.flake8` and `@pytest.mark.linter` markers for selective execution by category. This is a standard pattern in ROS 2 packages to ensure code style compliance is automatically enforced as part of continuous integration pipelines.

**Key concepts:** `PEP8 code style enforcement`, `ament_flake8 integration`, `pytest linter marker`, `automated style checking`, `CI pipeline linting`

## Sections

- **Flake8 Linter Test** — Describes a pytest test function that uses ament_flake8 to perform automated code style checks and fail on any PEP8 violations.
- **Exports** — Lists the exported test function `test_flake8()` which runs style checks and asserts no errors are present.
- **Dependencies** — Documents the required dependencies: ament_flake8 for linting integration and pytest for the test framework.
