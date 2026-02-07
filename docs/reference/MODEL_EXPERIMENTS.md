# Model Experiments Log

**Date:** 2026-01-30

## Important Context

**Critical Discrepancy:** All experiments below used **2-class classification** (clean/pit only). Production currently uses **3-class** (clean/maybe/pit) via a [two-stage training methodology](./TRAINING_METHODOLOGY.md) that has documented concerns:

- Stage 1: Binary classifier (pit vs no_pit)
- Stage 2: Fine-tune on misclassifications labeled "maybe"
- Result: 3-class output (clean/maybe/pit)

**This creates two problems:**
1. Our best experimental model (94.05% accuracy, 2-class) does not match production architecture
2. The two-stage approach used for production has safety risks and architectural issues

See [Training Methodology](./TRAINING_METHODOLOGY.md) for full assessment and recommended alternatives.

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

## Experiment Set 3: ResNet18 Backbone

**Date:** 2026-02-04
**Status:** Complete (GPU)

### Hypothesis
ResNet18 (11.7M params) can achieve similar accuracy to ResNet50 (25.6M params) with faster inference, making it suitable for deployment in latency-constrained environments.

### Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 (11.7M params vs 25.6M for ResNet50) |
| All layers trained | Yes |
| Augmentation | Enabled |
| Normalization | Disabled (matches production) |
| Epochs | 30 |
| Device | Tesla T4 GPU |
| Best Epoch | 6 |

### Results

| Model | Accuracy | Precision | Recall | F1 Score | Parameters | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Production (Baseline)** | 92.99% | - | - | 0.9293 | 25.6M | ~90MB |
| **ResNet50 Best** | 94.05% | - | - | 0.9397 | 25.6M | ~90MB |
| **ResNet18 (Exp 3)** | **91.92%** | 92.00% | 91.76% | 0.9186 | 11.7M | ~43MB |

### Per-Class Performance (ResNet18, Epoch 6)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| cherry_clean | 91.31% | 93.94% | 92.61% |
| cherry_pit | 92.69% | 89.58% | 91.11% |

### Training Curve Highlights
- Epoch 1: 87.19% (starting)
- Epoch 2: 89.72%
- Epoch 4: 90.13%
- Epoch 5: 90.86%
- **Epoch 6: 91.92%** (best)
- Epoch 7-30: Fluctuates 90-91.5% (slight overfitting)

### Analysis
*   **Accuracy trade-off:** ResNet18 achieves 91.92% vs 92.99% (production) - a 1.07% drop.
*   **Model size:** 43MB vs 90MB (2.1x smaller)
*   **Expected latency:** ~8-10ms vs ~16ms (estimated 40-50% faster based on FLOPs)
*   **Pit recall:** 89.58% - lower than clean recall (93.94%). For food safety, may want threshold adjustment.

### Recommendation
**Acceptable for deployment** if latency is a priority. The 1% accuracy drop is acceptable given:
- 2x smaller model (easier OTA updates)
- Faster inference (higher throughput or lower hardware requirements)
- Can be combined with threshold optimization to improve pit recall

### Artifacts
- Model: `training/experiments/resnet18_augmented_unnormalized/model_best.pt`
- Metrics: `training/experiments/resnet18_augmented_unnormalized/metrics.json`
- Config: `training/experiments/resnet18_augmented_unnormalized/config.yaml`
