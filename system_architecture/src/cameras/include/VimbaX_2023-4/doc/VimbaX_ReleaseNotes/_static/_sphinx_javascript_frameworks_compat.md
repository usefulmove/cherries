---
title: Sphinx JavaScript Frameworks Compatibility Shim
description: Provides backward compatibility utilities for jQuery and underscore.js,
  including URL encoding/decoding, query parameter parsing, text highlighting, and
  browser detection for Sphinx documentation.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_ReleaseNotes/_static/_sphinx_javascript_frameworks_compat.js
tags:
- sphinx
- jquery
- compatibility
- javascript
- documentation
related: []
last_analyzed: '2026-03-09T07:59:26Z'
---

# Sphinx JavaScript Frameworks Compatibility Shim

This JavaScript file is a compatibility shim for Sphinx documentation that extends jQuery with utility functions commonly needed for documentation sites. It provides URL encoding/decoding helpers, query string parameter parsing, text highlighting functionality (with special handling for SVG elements), and backward compatibility for the deprecated jQuery.browser API. This file is part of the VimbaX SDK documentation's static assets.

**Key concepts:** `URL encoding/decoding`, `query parameter parsing`, `text highlighting with DOM manipulation`, `SVG text highlighting`, `browser detection via user agent`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `jQuery.urldecode(x)` | function | Helper function to URL decode strings, replacing '+' with spaces before decoding |
| `jQuery.urlencode` | function | Alias for encodeURIComponent to URL encode strings |
| `jQuery.getQueryParameters(s)` | function | Parses URL query parameters and returns an object with arrays of string values for each key |
| `jQuery.fn.highlightText(text, className)` | method | Highlights matching text within DOM elements by wrapping it in span elements with the specified class, with special handling for SVG elements |
| `jQuery.uaMatch(ua)` | function | Parses user agent string to detect browser type and version |
| `jQuery.browser` | other | Object containing browser detection flags for backward compatibility with removed jQuery.browser API |

## Dependencies

`jquery`
