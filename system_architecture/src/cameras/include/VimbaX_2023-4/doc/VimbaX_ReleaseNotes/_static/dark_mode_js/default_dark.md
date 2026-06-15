---
title: Dark Mode Default Theme Initializer
description: JavaScript module that sets dark theme as the default for the VimbaX
  documentation, persisting the preference in localStorage.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_ReleaseNotes/_static/dark_mode_js/default_dark.js
tags:
- dark-mode
- theme-management
- javascript
- documentation
related: []
last_analyzed: '2026-03-09T07:59:42Z'
---

# Dark Mode Default Theme Initializer

This JavaScript file initializes the documentation page with a dark theme by default. It checks localStorage for a previously saved theme preference - if 'dark' is found, it applies the dark theme; if no preference exists, it defaults to dark mode and saves this preference. The theme is applied by setting a 'data-theme' attribute on the document root element, which CSS can then use to style the page accordingly.

**Key concepts:** `localStorage persistence`, `theme initialization`, `DOM attribute manipulation`, `default dark theme`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `loadTheme()` | function | Checks localStorage for theme preference and sets dark mode as default if no preference exists, applying the theme via data-theme attribute on the document root |
