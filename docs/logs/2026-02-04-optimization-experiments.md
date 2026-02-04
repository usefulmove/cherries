# Session: Optimization Experiments (Colab GPU Run)
**Date:** 2026-02-04

## Overview
Ran optimization experiments on Google Colab with GPU (Tesla T4) to evaluate ResNet18 as a faster alternative to ResNet50 for cherry pit detection.

## Experiments Run

### Experiment 1: Differential Learning Rates
- **Status:** SKIPPED (configured via `SKIP_EXPERIMENT_1 = True`)
- **Reason:** Previous CPU-only run was inconclusive; not worth re-running given that all-layer training (94.05%) already beats FC-only (92.99%)

### Experiment 2: Threshold Optimization
- **Status:** SKIPPED
- **Reason:** Production model not available in Colab clone (missing `cherry_classification.pt` in cloned repo)
- **Note:** Will need to run locally or upload production model to Drive

### Experiment 3: ResNet18 Backbone
- **Status:** COMPLETE
- **Result:** 91.92% accuracy (best at epoch 6)
- **Training time:** ~5-10s per epoch on Tesla T4
- **Model saved:** `training/experiments/resnet18_augmented_unnormalized/model_best.pt`

## Key Results

| Model | Accuracy | Parameters | Model Size |
|-------|----------|------------|------------|
| Production (ResNet50) | 92.99% | 25.6M | ~90MB |
| Best Training (ResNet50) | 94.05% | 25.6M | ~90MB |
| **ResNet18 (New)** | **91.92%** | 11.7M | ~43MB |

## Decisions Made
1. **Accept ResNet18 result** as viable for deployment (1% accuracy drop acceptable for speed/size benefits)
2. **Skip Experiment 1** permanently (differential LR not promising for this task)
3. **Need to run threshold optimization locally** (production model not in Colab environment)

## Artifacts Modified
- `training/experiments/resnet18_augmented_unnormalized/model_best.pt` (extracted from Colab)
- `training/experiments/resnet18_augmented_unnormalized/metrics.json`
- `training/experiments/resnet18_augmented_unnormalized/config.yaml`
- `docs/reference/MODEL_EXPERIMENTS.md` (updated with ResNet18 results)
- `training/notebooks/colab_optimization_experiments.ipynb` (fixed final summary cell)

## Issues Encountered
1. **Final summary cell TypeError:** `best_val_acc` was `None` (skipped experiment) but format string tried `:.2%` - fixed with `is not None` check
2. **Production model missing in Colab:** The cloned `cherries` repo doesn't include the production model weights, so threshold optimization couldn't run

## Next Steps
1. Run threshold optimization locally on production model (92.99%)
2. Prepare recommendations for developer meeting
3. Consider deploying ResNet18 if latency is a priority
