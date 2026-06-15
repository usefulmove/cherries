---
title: Default Light Theme Loader
description: JavaScript module that initializes the page theme from localStorage,
  defaulting to light mode if no preference is stored.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_static/dark_mode_js/default_light.js
tags:
- theme-switcher
- dark-mode
- localstorage
- ui-preferences
related: []
last_analyzed: '2026-03-09T07:56:43Z'
---

# Default Light Theme Loader

This JavaScript file is part of the VimbaX documentation's dark mode functionality. It defines and immediately invokes a loadTheme function that checks localStorage for a saved theme preference. If a 'dark' theme is stored, it sets the data-theme attribute on the document root element to 'dark'. If no theme preference exists, it defaults to 'light' mode by both storing 'light' in localStorage and setting the data-theme attribute accordingly. This script runs on page load to ensure consistent theme display across page navigation.

**Key concepts:** `theme persistence`, `localStorage`, `data-theme attribute`, `light mode default`, `DOM manipulation`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `const loadTheme = () => { ... }` | function | Loads and applies the user's theme preference from localStorage, defaulting to light mode if no preference exists |
