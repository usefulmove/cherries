---
title: Copyright Linter Test
description: Documentation for a pytest-based test that verifies copyright headers
  are present in source files using the ament_copyright linter, currently skipped
  due to missing headers in generated files.
source: docs/system_architecture/fanuc_comms/test/test_copyright.md
tags:
- testing
- linting
- copyright
- ament
- ros2
related: []
last_analyzed: '2026-03-09T07:44:13Z'
---

# Copyright Linter Test

This document describes a pytest test function called `test_copyright` that uses the ament_copyright linter to verify that all source and test files in the fanuc_comms package contain valid copyright headers. The test is currently decorated with `@pytest.mark.skip` because generated source files do not yet include copyright headers. When enabled, the test invokes `ament_copyright.main.main` with directory arguments `['.', 'test']` and asserts a zero return code to indicate no linting errors. The test also uses `@pytest.mark.copyright` and `@pytest.mark.linter` markers for categorization and selective test execution.

**Key concepts:** `copyright header validation`, `ament_copyright linter`, `pytest markers`, `skipped test`, `code compliance`

## Sections

- **Copyright Linter Test** — Describes the test_copyright pytest function that validates copyright headers using ament_copyright, currently skipped pending addition of headers to generated files.
- **Exports** — Lists the exported test_copyright function that runs the ament_copyright linter on package source and test directories.
- **Dependencies** — Documents the test's dependencies on ament_copyright and pytest packages.
