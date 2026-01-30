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

| Model | Variant | Accuracy | F1 Score | Latency (CPU) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | Production (Baseline) | 92.99% | 0.9293 | ~16 ms | **Baseline** |
| **ResNet50** | Augmented (Norm) | 93.96% | 0.9392 | ~37 ms | Archived |
| **ResNet50** | Scheduler (Norm) | 93.80% | 0.9375 | ~34 ms | Archived |
| **MobileNetV3** | Large (Norm) | 91.35% | 0.9132 | ~18 ms | Archived |
| **EfficientNet** | B0 (Norm) | 92.66% | 0.9259 | ~25 ms | Archived |
| **ResNet50** | **Augmented (Unnorm, Fixed)** | **94.05%** | **0.9397** | **~16.7 ms** | **Best so far** |


### Analysis
*   **Data Augmentation works:** Adding Rotation and Affine transforms improved ResNet50 accuracy.
*   **Unnormalized Training:** Training with **Augmentation** directly on 0-255 input range (matching production) yielded the best results (94.05%).
*   **Latency Optimization:** We identified and fixed a "denormal" value issue that initially caused 10x latency regression. The fixed unnormalized model matches production speed.
