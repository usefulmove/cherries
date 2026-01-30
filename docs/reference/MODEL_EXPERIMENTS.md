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
| **Production (Baseline)** | 92.99% | 0.9293 | ~16ms/image | Baseline |
| **ResNet50 Augmented** | **93.96%** | **0.9392** | ~37ms/image | Good Improvement (+0.97%) |
| **ResNet50 Scheduler** | 93.80% | 0.9375 | ~34ms/image | Good (+0.81%) |
| **MobileNetV3 Large** | 91.35% | 0.9132 | **~18ms/image** | Fast but less accurate (-1.64%) |
| **EfficientNet B0** | 92.66% | 0.9259 | ~25ms/image | Regression (-0.33%) |

## Experiment Set 2: Unnormalized Training & Optimization (2026-01-30)

We trained a "drop-in replacement" model to avoid code changes in the ROS2 node (which currently sends 0-255 raw images).

### Configurations
5. **ResNet50 Augmented Unnormalized**: ResNet50 + Augmentation, but trained/evaluated on 0-255 input range (No ImageNet normalization).

### Results
| Model | Accuracy | F1 Score | Inference Speed (CPU) | Result |
|-------|----------|----------|-----------------------|--------|
| **ResNet50 Unnormalized** | **94.05%** | **0.9397** | **~16.7ms/image*** | **FINAL WINNER** (+1.06%) |

### *The "Denormal" Latency Issue
Initial evaluation of the unnormalized model showed severe latency regression (**168ms** vs 16ms).
- **Cause:** The model weights contained "denormal" floating point numbers (extremely small values near zero but non-zero). CPUs handle these in software, causing massive slowdowns.
- **Fix:** We wrote a script (`fix_denormals.py`) to zero-out values < 1e-35.
- **Outcome:** Latency returned to ~16.7ms with no loss in accuracy.

### Next Steps
*   Deploy `resnet50_augmented_unnormalized` (fixed version) to the cherry system.
