# STORY-002: Model Optimization & Hyperparameter Tuning

## Goal
Improve the cherry pit detection model accuracy beyond the current baseline (~93%) by implementing data augmentation, hyperparameter tuning, and exploring alternative model architectures.

## Acceptance Criteria
- [x] Train a model with `augmentation: true` that exceeds 93% validation accuracy.
- [x] Evaluate at least 2 different hyperparameter configurations (learning rate, scheduler).
- [x] (Optional) Compare ResNet50 with a lighter backbone (e.g., MobileNetV3) for inference speed speedup.
- [x] Document final performance comparison table.

## Context Scope
**Write:**
- training/configs/
- training/notebooks/
- docs/reference/model_experiments.md

**Read:**
- training/README.md
- docs/core/PRD.md

## Approach
1. **Data Augmentation:** Enable `augmentation: true` in config (RandomHorizontalFlip, RandomVerticalFlip, ColorJitter).
2. **Hyperparameter Tuning:** Experiment with:
    - Learning Rate (currently 1e-4)
    - LR Scheduler (StepLR vs CosineAnnealing)
    - Batch Size
3. **Model Variations:**
    - Train MobileNetV3-Large for speed comparison.
    - Train EfficientNet-B0 for accuracy/efficiency trade-off.

## Notes
- **Final Winner:** `ResNet50 Augmented` achieved **93.96%** accuracy.
- **Baseline ResNet50 (Normalized) Accuracy:** 92.58%
- **Current Production (Raw) Accuracy:** 92.99%
- **MobileNetV3:** 91.35% (Faster but less accurate)
