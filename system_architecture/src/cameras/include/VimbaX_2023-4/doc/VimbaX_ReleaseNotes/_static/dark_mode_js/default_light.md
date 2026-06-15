---
title: Default Light Theme Loader
description: JavaScript module that loads and applies theme preference from localStorage,
  defaulting to light mode if no preference is set.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_ReleaseNotes/_static/dark_mode_js/default_light.js
tags:
- dark-mode
- theme-toggle
- localstorage
- documentation-ui
related: []
last_analyzed: '2026-03-09T07:59:48Z'
---

# Default Light Theme Loader

This JavaScript file implements theme loading functionality for VimbaX documentation. It defines a `loadTheme` function that checks localStorage for a saved theme preference. If a 'dark' theme is stored, it applies the dark theme via a data-theme attribute on the document root. If no preference exists, it defaults to light mode by both storing 'light' in localStorage and setting the corresponding data-theme attribute. The function is immediately invoked when the script loads.

**Key concepts:** `theme persistence`, `localStorage`, `data-theme attribute`, `default light mode`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `loadTheme()` | function | Loads theme preference from localStorage and applies it to the document. Defaults to light theme if no preference is stored. |
