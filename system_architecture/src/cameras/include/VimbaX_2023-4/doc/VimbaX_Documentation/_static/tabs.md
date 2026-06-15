---
title: VimbaX Documentation Sphinx Tabs Stylesheet
description: CSS stylesheet defining the visual appearance of tabbed navigation components
  in the VimbaX camera SDK documentation.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_static/tabs.css
tags:
- css
- sphinx
- documentation
- vimbax
- ui-styling
related: []
last_analyzed: '2026-03-09T07:57:15Z'
---

# VimbaX Documentation Sphinx Tabs Stylesheet

This CSS file provides styling for the sphinx-tabs extension used in the VimbaX 2023-4 documentation. It defines visual styles for tab containers, tab buttons, and tab panels with a red accent color (#e11021) that likely matches the VimbaX/Allied Vision brand. The stylesheet includes support for both light and dark themes, with the dark theme styles implemented using both CSS media queries for automatic theme detection (prefers-color-scheme) and explicit data-theme attributes. The styling uses the Lato font family with fallbacks to system fonts, and includes proper accessibility considerations with aria-selected states and focus indicators.

**Key concepts:** `sphinx-tabs styling`, `tab panel layouts`, `dark theme support`, `responsive color schemes`, `tablist accessibility attributes`

## Sections

- **Base tab styles** — Core styling for sphinx-tabs container, tablist, tab buttons, and panels
- **Dark theme (auto)** — CSS media query for automatic dark theme detection based on system preferences
- **Dark theme (explicit)** — Explicit dark theme styling using data-theme='dark' attribute
