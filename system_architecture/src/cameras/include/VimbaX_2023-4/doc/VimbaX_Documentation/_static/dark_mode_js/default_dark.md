---
title: Default Dark Mode Theme Loader
description: A small JavaScript utility that sets dark mode as the default theme for
  VimbaX documentation, persisting the user's theme preference in localStorage.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_static/dark_mode_js/default_dark.js
tags:
- dark-mode
- theme
- documentation
- vimbax
- frontend
related: []
last_analyzed: '2026-03-09T07:56:42Z'
---

# Default Dark Mode Theme Loader

This JavaScript file implements a simple theme loader for VimbaX documentation that defaults to dark mode. It checks localStorage for a saved theme preference - if one exists and is set to 'dark', it applies the dark theme. If no preference exists, it sets 'dark' as the default theme in both localStorage and the document. The theme is applied by setting a 'data-theme' attribute on the document's root element.

**Key concepts:** `localStorage persistence`, `theme management`, `DOM attribute manipulation`, `default dark theme`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `loadTheme()` | function | Loads and applies the theme from localStorage, defaulting to dark mode if no preference is stored |
