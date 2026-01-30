# Model Experiments Log

**Date:** 2026-01-30

## Experiment Set 1: Augmentation & Architecture Search

We ran 4 experiments to beat the production baseline (92.99% accuracy).

### Configurations
1.  **ResNet50 Augmented**: ResNet50 + Rotation/Affine/Jitter.
2.  **ResNet50 Scheduler**: Same as above + Cosine Annealing LR.
3.  **MobileNetV3 Large**: Lightweight model + Augmentation + Scheduler.
4.  **EfficientNet B0**: Efficient model + Augmentation + Scheduler.

### Results Benchmark

We evaluated all models on the validation set (N=1226).

| Model | Accuracy | F1 Score | Inference Speed (CPU) | Result |
|-------|----------|----------|-----------------------|--------|
| **Production (Baseline)** | 92.99% | 0.9293 | ~1.42 batch/s | Baseline |
| **ResNet50 Augmented** | **93.96%** | **0.9392** | ~3.68 s/batch | **WINNER** (+0.97%) |
| **ResNet50 Scheduler** | 93.80% | 0.9375 | ~3.35 s/batch | Good (+0.81%) |
| **MobileNetV3 Large** | 91.35% | 0.9132 | **~1.83 s/batch** | Fast but less accurate (-1.64%) |

### Analysis
*   **Data Augmentation works:** Adding Rotation and Affine transforms improved ResNet50 accuracy from ~92.6% (previous baseline) to **93.96%**.
*   **MobileNetV3 is faster but weaker:** It is ~2x faster than ResNet50 on CPU, but dropped accuracy by ~1.6%.
*   **Winner:** `resnet50_augmented` is the best performing model.

### Next Steps
*   Deploy `resnet50_augmented` to the cherry system.
