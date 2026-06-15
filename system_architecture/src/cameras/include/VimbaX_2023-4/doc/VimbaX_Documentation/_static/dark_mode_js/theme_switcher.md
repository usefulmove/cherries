---
title: Sphinx RTD Dark Mode Theme Switcher
description: JavaScript module that creates a dark/light theme toggle button for Sphinx
  Read the Docs documentation, persisting user preference via localStorage.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_static/dark_mode_js/theme_switcher.js
tags:
- theme-switcher
- dark-mode
- sphinx-documentation
- frontend
- user-preference
related: []
last_analyzed: '2026-03-09T07:56:47Z'
---

# Sphinx RTD Dark Mode Theme Switcher

This JavaScript file implements a theme switcher for Sphinx Read the Docs documentation pages, allowing users to toggle between light and dark modes. It creates a button with sun and moon icons using Font Awesome, stores the user's theme preference in localStorage, and applies the theme by setting a data-theme attribute on the document element. The module uses jQuery for DOM ready handling, click events, and fade animations. It also appends attribution to the footer crediting the sphinx_rtd_dark_mode project by MrDogeBro.

**Key concepts:** `localStorage theme persistence`, `DOM manipulation for theme toggle`, `jQuery animations`, `data-theme attribute switching`, `Font Awesome icons`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `const createThemeSwitcher = () => {...}` | function | Creates and appends a theme switcher button with moon/sun icons to the document body, initializing visibility based on stored theme preference |
| `const switchTheme = () => {...}` | function | Toggles between dark and light themes, updating localStorage, data-theme attribute, and animating icon visibility changes |

## Dependencies

`jquery`, `font-awesome`
