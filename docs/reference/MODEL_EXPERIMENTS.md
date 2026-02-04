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

---

## Experiment Set 2: Differential Learning Rates (CPU - Inconclusive)

**Date:** 2026-02-04

**Warning:** This experiment ran on **CPU only** due to misconfigured Colab runtime. Results are not representative and the experiment should be considered inconclusive.

### Hypothesis
Freezing early layers (layer1-3) and using differential learning rates for later layers (layer4, fc) might improve accuracy by preserving generic ImageNet features while adapting task-specific layers more aggressively.

### Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | ResNet50 |
| Frozen Layers | layer1, layer2, layer3 |
| layer4 LR | 1e-5 |
| fc LR | 1e-3 |
| Augmentation | Enabled |
| Normalization | Disabled (matches production) |
| Epochs | 30 |
| Device | **CPU** (no GPU - misconfiguration) |

### Results

| Model | Accuracy | F1 Score | Notes |
| :--- | :--- | :--- | :--- |
| **Production (Baseline)** | 92.99% | 0.9293 | FC-only training, frozen backbone |
| **Previous Best** | 94.05% | 0.9397 | All layers trained |
| **Differential LR (CPU)** | 88.42% | 0.8825 | **Inconclusive - CPU only** |

### Analysis
*   **Result:** 88.42% accuracy - significantly worse than baseline (92.99%) and previous best (94.05%).
*   **Caveats:** 
    - Training ran on CPU (~12 min/epoch vs ~30s on GPU)
    - CPU training may have different numerical behavior
    - Results cannot be directly compared to GPU-trained models
*   **Observation:** The original production model only trained the FC layer (92.99%). Our best model trains ALL layers (94.05%). This suggests cherry pit detection benefits from adapting early features, not freezing them.
*   **Threshold Analysis:** Saved to `training/experiments/resnet50_differential_lr_cpu/threshold_analysis/` but not actionable given inconclusive base accuracy.

### Recommendation
Do not pursue differential LR approach further. The evidence suggests training all layers (current best approach) is more effective for this task. Focus on:
1. ResNet18 backbone (smaller/faster, similar accuracy potential)
2. Threshold optimization on the existing 94.05% model
3. Collecting more recent training data if available

---

## Experiment Set 3: ResNet18 Backbone (Pending)

**Status:** Not yet run. ResNet18 experiment did not complete due to Colab timeout (CPU training was too slow).

### Planned Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 (11.7M params vs 25.6M for ResNet50) |
| All layers trained | Yes |
| Augmentation | Enabled |
| Normalization | Disabled |
| Expected Latency | ~8-10ms (vs 16.7ms for ResNet50) |

### Next Steps
1. Re-run on Colab with **GPU runtime enabled**
2. Compare accuracy vs ResNet50
3. If accuracy is within 1-2% of ResNet50, consider for production (faster inference)
