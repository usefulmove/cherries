---
title: VimbaX Dark Mode General CSS Styles
description: CSS stylesheet providing general styles for dark mode theme switching
  in VimbaX documentation, including input field styling and theme switcher button
  positioning.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_ReleaseNotes/_static/dark_mode_css/general.css
tags:
- css
- dark-mode
- documentation
- sphinx
- vimbax
related: []
last_analyzed: '2026-03-09T07:59:41Z'
---

# VimbaX Dark Mode General CSS Styles

This CSS file is part of the VimbaX 2023-4 SDK documentation's dark mode implementation. It defines styles for form input elements (removing box shadows), creates a circular fixed-position theme switcher button in the bottom-right corner of the page, and sets up CSS transitions for smooth theme switching across various documentation elements including navigation, content areas, code blocks, and buttons. The selectors target Read the Docs/Sphinx theme-specific classes (wy-nav, rst-content) to ensure compatibility with the documentation framework.

**Key concepts:** `dark mode theme switching`, `CSS transitions`, `input field styling`, `fixed position theme switcher button`, `Read the Docs theme compatibility`

## Sections

- **Input field styles** — Removes box shadows from various form input types
- **Theme switcher button** — Styles for the fixed-position circular theme toggle button
- **Transition definitions** — Defines transition animations for multiple page elements during theme switching
