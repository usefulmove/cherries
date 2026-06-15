---
title: Flake8 Linter Test
description: A pytest-based test that runs flake8 code style checks on the image_service
  package using ament_flake8, asserting zero style errors or warnings.
source: docs/system_architecture/image_service/test/test_flake8.md
tags:
- linting
- flake8
- code-style
- testing
- ros2
related: []
last_analyzed: '2026-03-09T07:44:53Z'
---

# Flake8 Linter Test

This documentation describes a pytest test function, `test_flake8`, that uses `ament_flake8`'s `main_with_errors` to lint the entire image_service package for PEP8 and flake8 style violations. The test is decorated with `@pytest.mark.flake8` and `@pytest.mark.linter` markers, allowing it to be selectively run as part of a linter test suite, which is a common pattern in ROS 2 / ament-based projects. If any style errors or warnings are found, the test fails and prints all discovered violations. This file exists to enforce consistent code style as part of continuous integration.

**Key concepts:** `ament_flake8 integration`, `PEP8 / code style enforcement`, `pytest linter marker`, `automated style gate`

## Sections

- **Flake8 Linter Test** — Introduces the pytest test function that runs ament_flake8 to enforce PEP8 style compliance on the image_service package.
- **Exports** — Documents the exported `test_flake8()` function that runs ament_flake8 and asserts zero return code for style compliance.
- **Dependencies** — Lists the dependencies required for this test: ament_flake8 and pytest.
