# Session: Model Optimization & Benchmarking
**Date:** 2026-01-30

## Overview
Successfully executed a batch of experiments to improve the cherry pit classifier. We established a rigorous benchmarking pipeline and identified a model that outperforms the production baseline.

## Key Decisions
- **Augmentation is Key:** Adopted `RandomRotation`, `RandomAffine`, and `ColorJitter`. This was the primary driver of accuracy improvement.
- **Rejected MobileNetV3:** Despite theoretical efficiency, it was less accurate (-1.6%) and slower on our CPU benchmark than ResNet50.
- **Rejected Scheduler-only approach:** While better than baseline, it performed slightly worse than the augmented model without scheduler.
- **Normalization Discrepancy:** Discovered the production system uses raw images (0-255), while standard training uses ImageNet normalization. This requires either changing the production code or training a specialized unnormalized model.

## Artifacts Modified
- **Code:**
  - `training/src/data.py`: Added comprehensive transforms.
  - `training/src/model.py`: Refactored for multiple architectures.
  - `training/scripts/compare_models.py`: Updated to support architecture flags and factory pattern.
  - `training/notebooks/colab_experiments.ipynb`: Created for batch execution.
- **Configs:** Created `resnet50_augmented.yaml`, `resnet50_scheduler.yaml`, `mobilenetv3_large.yaml`, `efficientnet_b0.yaml`.
- **Docs:**
  - `docs/stories/STORY-002-Model-Optimization.md`: Updated with results.
  - `docs/reference/model_experiments.md`: Created detailed benchmark log.

## Results
| Model | Accuracy | Result |
|-------|----------|--------|
| Production (Baseline) | 92.99% | Baseline |
| **ResNet50 Augmented** | **93.96%** | **Winner (+0.97%)** |
| MobileNetV3 Large | 91.35% | Failed |

## Open Items
- **Deployment:** The winning model requires the inference node to be updated to support ImageNet normalization.
- **Unnormalized Experiment:** User suggested training a model *without* normalization (but with augmentation) to allow drop-in replacement without code changes.

## Next Steps
- Archive STORY-002.
- Evaluate the "Unnormalized Training" path before full deployment.
