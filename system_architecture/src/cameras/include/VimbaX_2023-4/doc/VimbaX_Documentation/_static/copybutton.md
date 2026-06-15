---
title: Copy Button CSS Styles
description: CSS stylesheet that provides styling for copy-to-clipboard buttons used
  in code highlighting blocks within VimbaX documentation.
source: src/cameras/include/VimbaX_2023-4/doc/VimbaX_Documentation/_static/copybutton.css
tags:
- css
- documentation
- copy-button
- tooltip
- vimbax
related: []
last_analyzed: '2026-03-09T07:56:15Z'
---

# Copy Button CSS Styles

This CSS file defines styles for copy-to-clipboard buttons (copybtn) that appear on code highlighting blocks in the VimbaX documentation. The buttons are positioned absolutely in the top-right corner of code blocks, hidden by default and revealed on hover with smooth opacity transitions. The styling follows GitHub's color conventions. The file also includes a minimal CSS-only tooltip implementation for left-positioned tooltips using the data-tooltip attribute. A print media query ensures copy buttons are hidden when printing the documentation pages.

**Key concepts:** `copy button styling`, `hover opacity transitions`, `GitHub-style color scheme`, `CSS-only tooltips`, `print media query`, `code highlight blocks`

## Sections

- **Copy Button Styles** — Main button styling including positioning, colors, hover states, and success states
- **CSS Tooltip** — Minimal CSS-only tooltip implementation for left-positioned tooltips
- **Print Media Query** — Hides copy buttons when printing
