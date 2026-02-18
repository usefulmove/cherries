---
type: framework_index
framework: PyTorch
description: Comprehensive PyTorch documentation and tutorials for the Cherry Processing System.
---

# PyTorch Framework Documentation

## Overview

This directory contains structured PyTorch documentation covering the complete machine learning pipeline for cherry processing—from tensor basics to production deployment.

## Documentation Index

### Getting Started
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [01. Overview](PYTORCH_01_OVERVIEW.md) | System architecture, two-stage pipeline | None |
| [02. Models](PYTORCH_02_MODELS.md) | Model definitions, loading patterns | 01 |

### Core Concepts
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [03. GPU Tensors](PYTORCH_03_GPU_TENSORS.md) | Device management, tensor operations | 01-02 |
| [04. Preprocessing](PYTORCH_04_PREPROCESSING.md) | Image transforms, data loading | 01-03 |
| [05. Inference](PYTORCH_05_INFERENCE.md) | Model evaluation, prediction patterns | 01-04 |

### Advanced Topics
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [06. Functional API](PYTORCH_06_FUNCTIONAL.md) | Functional programming, nn.functional | 01-05 |
| [07. Training](PYTORCH_07_TRAINING.md) | Loss functions, optimizers, loops | 01-06 |
| [08. Postprocessing](PYTORCH_08_POSTPROCESSING.md) | Results handling, metrics | 01-07 |
| [09. Complete Pipeline](PYTORCH_09_COMPLETE_PIPELINE.md) | End-to-end integration | 01-08 |

## Quick Reference

### Common Patterns
```python
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model loading
model = torch.load('model.pt', map_location=device)
model.eval()

# Inference
def predict(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        return output.cpu()
```

### Key Code Locations
- `threading_ws/src/cherry_detection/cherry_detection/ai_detector3.py` - Main inference implementation (v6/hdr_v1)
- `training/scripts/` - Training and evaluation utilities

## Related Resources
- [System Architecture](../../architecture/inference_pipeline/ARCHITECTURE.md)
- [Training Colab Skill](../../../skills/training-colab/SKILL.md)
- [Model Evaluation Skill](../../../skills/evaluate-model/SKILL.md)
