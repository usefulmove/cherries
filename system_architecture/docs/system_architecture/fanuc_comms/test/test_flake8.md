---
title: Flake8 Linter Test
description: A pytest-based linter test that runs flake8 code style checks on the
  fanuc_comms package using the ament_flake8 integration.
source: docs/system_architecture/fanuc_comms/test/test_flake8.md
tags:
- testing
- linting
- code-style
- flake8
- ros2
related: []
last_analyzed: '2026-03-09T07:44:16Z'
---

# Flake8 Linter Test

This documentation describes a pytest test function that invokes the ament_flake8 linter (an ament-flavored wrapper around flake8) against the fanuc_comms package. The test asserts that no code style violations are found and surfaces any style errors or warnings as the assertion message on failure. It uses pytest markers (@pytest.mark.flake8 and @pytest.mark.linter) to integrate with the ROS 2 / ament test infrastructure for selective test execution. This pattern is standard boilerplate in ROS 2 packages for enforcing consistent Python code style.

**Key concepts:** `ament_flake8 integration`, `PEP8 / code style enforcement`, `pytest markers for linting`, `automated code quality checks`, `ROS 2 test infrastructure`

## Sections

- **Flake8 Linter Test** — Describes the pytest test function that runs ament_flake8 linter checks on the package and asserts zero code style violations.
- **Exports** — Lists the test_flake8 function export with its pytest marker decorations for test infrastructure filtering.
- **Dependencies** — Lists the required dependencies: ament_flake8 and pytest.
