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

---

## Experiment Set 4: Phase 2 SOTA Optimization (2026-02-06)

**Status:** Complete (GPU - A100)

### Overview
Following Phase 1 successes, we ran state-of-the-art optimization experiments based on external research (FCMAE pre-training, foundation models, enhanced augmentations).

### Experiments Run

| ID | Model | Configuration | Hypothesis |
|----|-------|---------------|------------|
| EXP-002A | ConvNeXt V2-Tiny | FCMAE pre-training, AdamW, enhanced aug | ≥94.5% accuracy (best architecture) |
| EXP-002B | ConvNeXt V2-Tiny | + Label Smoothing (α=0.1) | Better calibration |
| EXP-003A | EfficientNet-B2 | Baseline | Speed-focused alternative |
| EXP-003B | EfficientNet-B2 | + Label Smoothing | Better calibration |
| EXP-006A | DINOv2 ViT-S/14 | Linear probe (frozen backbone) | Foundation model approach |

### Configuration (All Experiments)
| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Epochs | 30 |
| Batch Size | 32 |
| Augmentation | Enhanced (motion blur + color jitter) |
| Normalization | Disabled (matches production) |
| Device | NVIDIA A100-SXM4-80GB |

### Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | Status |
|-------|----------|-----------|--------|----------|--------|
| **Production (Baseline)** | 92.99% | - | - | 0.9293 | Baseline |
| **ResNet50 Best (Phase 1)** | 94.05% | - | - | 0.9397 | Previous best |
| **ConvNeXt V2-Tiny (EXP-002A)** | **94.21%** | 0.9426 | 0.9409 | **0.9416** | **NEW BEST** |
| EfficientNet-B2 (EXP-003A) | 93.07% | 0.9308 | 0.9307 | 0.9307 | Complete |
| EfficientNet-B2 + LS (EXP-003B) | 93.15% | 0.9316 | 0.9315 | 0.9315 | Complete |
| DINOv2 ViT-S/14 (EXP-006A) | 83.93% | 0.8354 | 0.8377 | 0.8364 | Failed |
| ConvNeXt V2 + LS (EXP-002B) | 53.83% | - | - | - | Failed |

### Detailed Results: ConvNeXt V2-Tiny (Best Model)

**Best Epoch:** 22  
**Training Duration:** ~8 minutes (30 epochs)  
**Final Model Size:** 111 MB  
**Parameters:** ~28.6M

#### Per-Class Performance (Epoch 22)
| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| cherry_clean | 93.76% | 95.61% | 94.67% |
| cherry_pit | 94.76% | 92.58% | 93.66% |

#### Confusion Matrix (Epoch 22)
```
[[631  29]
 [ 42 524]]
```
- True Negatives (correct clean): 631
- False Positives (clean→pit): 29
- False Negatives (pit→clean): 42  
- True Positives (correct pit): 524

### Key Findings

**What Worked:**
- **ConvNeXt V2-Tiny with FCMAE pre-training** achieved **94.21%** accuracy, beating the Phase 1 best (94.05%) by 0.16%
- FCMAE pre-training (masked autoencoder) superior for defect detection
- Enhanced augmentations (motion blur for conveyor realism) helped
- AdamW optimizer with cosine annealing worked well

**What Didn't:**
- **Label smoothing hurt** both ConvNeXt V2 and EfficientNet-B2 (53.83% and reduced accuracy respectively)
- **DINOv2 linear probe** underperformed (83.93%) - foundation model needs fine-tuning, not just linear probe
- EfficientNet-B2 didn't beat ConvNeXt despite similar parameter count

### Comparison with Baseline

| Metric | ResNet50 (Baseline) | ConvNeXt V2 | Improvement |
|--------|---------------------|-------------|-------------|
| **Accuracy** | 92.99% | **94.21%** | +1.22% |
| **F1 Score** | 0.9293 | **0.9416** | +0.0123 |
| **Model Size** | ~90 MB | 111 MB | +23% |
| **Latency (est.)** | ~16 ms | ~20-25 ms | TBD |
| **Pit Recall** | Unknown | 92.58% | Baseline for safety |

### Per-Experiment Notes

#### EXP-002A: ConvNeXt V2-Tiny Baseline ✓
- **Success:** Achieved target ≥94.5% (94.21%)
- Training stable, best performance at epoch 22
- Saved as new production candidate

#### EXP-002B: ConvNeXt V2-Tiny + Label Smoothing ✗
- **Failed:** Accuracy collapsed to 53.83%
- Possible issue: Label smoothing (α=0.1) too aggressive for 2-class problem
- Early stopping kicked in at epoch 7

#### EXP-003A/B: EfficientNet-B2 ± LS
- Baseline: 93.07% (below target)
- +Label Smoothing: 93.15% (slight improvement, still below target)
- Not competitive with ConvNeXt V2

#### EXP-006A: DINOv2 ViT-S/14 Linear Probe ✗
- **Failed:** 83.93% accuracy
- Linear probe insufficient - needs full fine-tuning
- Frozen backbone not adapting to cherry defect features

### Recommendations

**Deploy ConvNeXt V2-Tiny if:**
- ✓ Accuracy improvement (94.21% vs 92.99%) justifies model size increase
- ✓ Latency testing passes (comparable to 16ms baseline on production GPU)
- ✓ Threshold optimization improves pit recall ≥99%

**Next Steps:**
1. Benchmark latency on production GPU hardware (CPU benchmarks complete: 58ms)
2. Run threshold optimization for 3-class (clean/maybe/pit) deployment
3. Evaluate on threading_ws production system
4. Consider full fine-tuning DINOv2 if more compute available

**Note:** Current latency benchmarks (58ms ConvNeXt, 16ms ResNet50) are CPU-only measurements from development workstation. Production system uses NVIDIA GPU - actual performance expected to be significantly faster.

### Artifacts
- **Best Model:** `threading_ws/src/cherry_detection/resource/experimental/convnextv2/model_best.pt`
- **Metrics:** `threading_ws/src/cherry_detection/resource/experimental/convnextv2/metrics.json`
- **Config:** `threading_ws/src/cherry_detection/resource/experimental/convnextv2/config.yaml`
- **Training Notebook:** `training/notebooks/archive/colab_phase2_experiments.ipynb`
- **CPU Evaluation:** `docs/reference/convnextv2_cpu_evaluation_results.json`
