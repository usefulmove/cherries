---
title: VmbCPP StringFeature Class Definition
description: Header file defining the StringFeature class for handling string-type
  camera features in the VimbaX C++ API.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/StringFeature.h
tags:
- vimbax
- camera-api
- cpp-header
- feature-handling
- allied-vision
related: []
last_analyzed: '2026-03-09T07:55:19Z'
---

# VmbCPP StringFeature Class Definition

This header file defines the StringFeature class within the VmbCPP namespace, which is part of Allied Vision's VimbaX camera SDK. StringFeature extends BaseFeature and provides specialized implementation for string-type camera features. The class includes a constructor that takes feature info and a container reference, along with protected virtual methods SetValue and GetValue for managing string data. The GetValue method is specifically designed to handle data passing across DLL boundaries using a buffer and length pattern. This is an internal implementation class intended for use within the VmbCPP library.

**Key concepts:** `StringFeature class`, `BaseFeature inheritance`, `VmbErrorType error handling`, `Feature container pattern`, `DLL boundary data passing`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class StringFeature final : public BaseFeature` | class | A final class that handles string-type features in the VimbaX camera API, inheriting from BaseFeature |
| `StringFeature(const VmbFeatureInfo& featureInfo, FeatureContainer& featureContainer)` | method | Constructor that initializes a string feature with feature info and a reference to its container |
| `virtual VmbErrorType SetValue(const char *pValue) noexcept override` | method | Protected virtual method to set the string feature value |
| `virtual VmbErrorType GetValue(char * const pValue, VmbUint32_t &length) const noexcept override` | method | Protected virtual method to get the string feature value, designed for DLL boundary data passing |
