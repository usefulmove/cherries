# Session: Unnormalized Model Evaluation
**Date:** 2026-01-30

## Overview
Evaluated the `resnet50_augmented_unnormalized` model against the production baseline. Identified and resolved a critical latency regression caused by denormal floating-point values in the model weights.

## Key Decisions
- **Validation Success:** The unnormalized model achieved **94.05% Accuracy**, beating the production baseline of 92.99%.
- **Latency Fix:** Initial inference was ~10x slower (168ms). Diagnosed as CPU handling of denormal numbers. Created `fix_denormals.py` to sanitize weights.
- **Final Performance:** Post-fix latency is **16.7ms**, matching production performance (<30ms).

## Artifacts Modified
- `docs/stories/STORY-003-Deployment-Readiness.md`: Updated with evaluation results.
- `training/scripts/benchmark_latency.py`: Created for pure compute testing.
- `training/scripts/fix_denormals.py`: Created to fix model weights.
- `training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt`: The finalized, high-performance model file.

## Open Items
- Deployment of `model_best_fixed.pt` to `cherry_system/cherry_detection/resource/cherry_classification.pt` is pending user approval.
