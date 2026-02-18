---
name: evaluate-model
description: Compare a trained model against the production baseline for accuracy and F1 score.
---

# Evaluate Model Skill

This skill allows you to compare a new candidate model against the currently deployed production model to verify improvements in accuracy, precision, recall, and F1 score.

## When to use
*   After training a new model.
*   Before deploying a model to production.
*   To verify if a model regression has occurred.

## Requirements
*   A trained PyTorch model file (`.pt`).
*   The production model file (typically `threading_ws/src/cherry_detection/resource/classification-2_26_2025-iter5.pt`).
*   Validation dataset located at `../cherry_classification/data` (or specified path).
*   Python environment with `torch`, `torchvision`, `numpy`, `tqdm`.

## Usage

Run the `compare_models.py` script from the repository root.

```bash
python training/scripts/compare_models.py \
  --new-model <path/to/new_model.pt> \
  --prod-model <path/to/production_model.pt> \
  [--unnormalized] \
  [--architecture <resnet50|mobilenet_v3_large|efficientnet_b0>]
```

### Options
*   `--new-model`: Path to the new candidate model.
*   `--prod-model`: Path to the baseline model (usually `threading_ws/src/cherry_detection/resource/classification-2_26_2025-iter5.pt`).
*   `--unnormalized`: **Important!** Use this flag if the new model was trained on 0-255 raw images (no ImageNet normalization). The production system typically uses unnormalized images.
*   `--architecture`: Model architecture (default: `resnet50`).
*   `--device`: `cpu` or `cuda` (default: auto-detect).

## Example

Evaluate a new unnormalized ResNet50 model:

```bash
python training/scripts/compare_models.py \
  --new-model training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt \
  --prod-model threading_ws/src/cherry_detection/resource/classification-2_26_2025-iter5.pt \
  --unnormalized
```

## Interpreting Results
The script prints a comparison table.
*   **Accuracy > 93%** is the typical target.
*   Ensure **F1 Score** does not regress.
*   Check per-class metrics if specific class performance (e.g., Pit recall) is critical.
