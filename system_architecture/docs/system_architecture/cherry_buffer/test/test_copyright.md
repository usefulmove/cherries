---
title: Copyright Linter Test
description: A pytest test documentation that describes the ament_copyright linter
  test for verifying proper copyright headers in the cherry_buffer package's source
  files.
source: docs/system_architecture/cherry_buffer/test/test_copyright.md
tags:
- testing
- linter
- copyright
- ament
- ros2
related: []
last_analyzed: '2026-03-09T07:42:32Z'
---

# Copyright Linter Test

This documentation describes a pytest test function called `test_copyright` that uses the ament_copyright linter tool to verify all source files in the cherry_buffer package contain proper copyright headers. The test checks both the main source directory (`.`) and the `test` directory. It is currently decorated with `@pytest.mark.skip` because generated source files in the package do not yet have copyright headers. The test is also tagged with `@pytest.mark.copyright` and `@pytest.mark.linter` markers for selective test execution, and asserts that the linter returns a zero exit code to indicate compliance.

**Key concepts:** `copyright header compliance`, `ament_copyright linter`, `pytest markers`, `skipped test`, `Apache 2.0 license`

## Sections

- **Copyright Linter Test** — Describes the test_copyright pytest function that invokes ament_copyright linter and explains why it is currently skipped due to missing copyright headers in generated files.
- **Exports** — Lists the test_copyright function as the single exported item that runs the ament_copyright linter and asserts a zero return code.
- **Dependencies** — Documents the external dependencies required by the test: ament_copyright and pytest.
