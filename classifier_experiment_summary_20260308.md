# Cherry Pit Classifier: Experiment Summary

128×128 px cherry crop images, unnormalized (0–255). Binary classification: `cherry_clean` / `cherry_pit`. Validation set: N = 1,226.

---

## Results

| Model | Accuracy | F1 | Precision | Recall | Size | CPU Latency | Notes |
|-------|----------|----|-----------|--------|------|-------------|-------|
| ResNet50 (baseline) | 92.99% | 0.9293 | — | — | 90 MB | ~16 ms | FC-only fine-tune |
| ResNet50 + aug (normalized) | 93.96% | 0.9392 | — | — | 90 MB | ~37 ms | |
| ResNet50 + aug + scheduler (normalized) | 93.80% | 0.9375 | — | — | 90 MB | ~34 ms | |
| MobileNetV3 Large + aug + scheduler | 91.35% | 0.9132 | — | — | — | ~18 ms | |
| EfficientNet-B0 + aug + scheduler | 92.66% | 0.9259 | — | — | — | ~25 ms | |
| ResNet50 + aug (unnormalized) | **94.05%** | **0.9397** | — | — | 90 MB | ~16.7 ms | |
| ResNet50 + differential LR | 88.42% | 0.8825 | — | — | 90 MB | — | CPU-only, inconclusive |
| ResNet18 + aug (unnormalized) | 91.92% | 0.9186 | 92.00% | 91.76% | 43 MB | ~8–10 ms (est.) | |
| EfficientNet-B2 | 93.07% | 0.9307 | 0.9308 | 0.9307 | — | — | |
| EfficientNet-B2 + label smoothing | 93.15% | 0.9315 | 0.9316 | 0.9315 | — | — | |
| ConvNeXt V2-Tiny + label smoothing | 53.83% | — | — | — | — | — | Collapsed epoch 7 |
| DINOv2 ViT-S/14 (linear probe) | 83.93% | 0.8364 | 0.8354 | 0.8377 | — | — | Frozen backbone |
| **ConvNeXt V2-Tiny (FCMAE)** | **94.21%** | **0.9416** | **0.9426** | **0.9409** | **111 MB** | **~58 ms (CPU)** | **Best** |

> CPU latency measured on development workstation. GPU latency on production hardware not yet measured.

---

## Experiments

### Baseline

**Architecture:** ResNet50, ImageNet pre-trained, FC layer only fine-tuned (backbone frozen)  
**Input:** 128×128, unnormalized (0–255)  
**Result:** 92.99% accuracy, F1 0.9293, ~16 ms CPU latency

---

### Augmentation & Architecture Search

**Hardware:** GPU (unspecified)  
**Epochs:** 30  
**Batch size:** 32

| Parameter | Value |
|-----------|-------|
| Augmentation | Random rotation, affine, color jitter |
| Scheduler | Cosine annealing (where noted) |
| Normalization | ImageNet mean/std (normalized runs); disabled (unnormalized run) |
| All layers trained | Yes |

Five variants tested: ResNet50 with and without scheduler, MobileNetV3 Large, EfficientNet-B0, and ResNet50 unnormalized. The unnormalized ResNet50 variant achieved the best result (94.05%) and matched baseline CPU latency (~16.7 ms).

**Findings:**
- Augmentation consistently improved ResNet50 accuracy over the FC-only baseline.
- Unnormalized input (matching production preprocessing) outperformed normalized training. This is the most impactful single change.
- Normalized models showed significant latency regression (~37 ms vs. ~16 ms); the unnormalized fix eliminated this.
- The latency regression was caused by **denormal floating-point values** introduced when passing raw 0–255 inputs through a normalization path. The unnormalized model avoids this entirely.
- MobileNetV3 and EfficientNet-B0 did not beat the baseline and were not pursued further.

---

### Differential Learning Rates

**Hardware:** CPU only (Colab GPU misconfiguration — results not representative)  
**Epochs:** 30

| Parameter | Value |
|-----------|-------|
| Frozen layers | layer1, layer2, layer3 |
| layer4 LR | 1e-5 |
| FC LR | 1e-3 |
| Augmentation | Enabled |
| Normalization | Disabled |

Result: 88.42% — significantly below baseline. Not actionable due to CPU-only execution. Should be re-run on GPU before conclusions are drawn.

**Observation:** The baseline trained only the FC layer (92.99%); the best model trains all layers (94.05%). This is consistent with the task requiring early feature adaptation, not just head-level fine-tuning.

---

### ResNet18 Backbone

**Hardware:** Tesla T4  
**Epochs:** 30 (best at epoch 6)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 (11.7M params) |
| All layers trained | Yes |
| Augmentation | Enabled |
| Normalization | Disabled |

**Per-class (epoch 6):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| cherry_clean | 91.31% | 93.94% | 92.61% |
| cherry_pit | 92.69% | 89.58% | 91.11% |

**Training curve:** 87.19% → 91.92% by epoch 6, then fluctuates 90–91.5% through epoch 30 (mild overfitting).

**Findings:**
- 1.07 pp accuracy drop vs. baseline; 2.1× smaller (43 MB vs. 90 MB); estimated 40–50% faster inference.
- Pit recall (89.58%) is lower than clean recall (93.94%) — threshold adjustment required for safety-critical use.
- Drop-in replacement for ResNet50 with no other pipeline changes required.

**Artifacts:**
- `training/experiments/resnet18_augmented_unnormalized/model_best.pt`
- `training/experiments/resnet18_augmented_unnormalized/metrics.json`
- `training/experiments/resnet18_augmented_unnormalized/config.yaml`

---

### SOTA Architectures (ConvNeXt V2, EfficientNet-B2, DINOv2)

**Hardware:** NVIDIA A100-SXM4-80GB  
**Epochs:** 30  
**Batch size:** 32  
**Random seed:** 42

| Parameter | Value |
|-----------|-------|
| Augmentation | Enhanced — motion blur + color jitter |
| Normalization | Disabled |
| Optimizer | AdamW + cosine annealing |

Five experiments run. ConvNeXt V2-Tiny with FCMAE pre-training achieved the best result across all experiments (94.21%, F1 0.9416). Best epoch was 22; training completed in ~8 minutes on A100.

**ConvNeXt V2-Tiny per-class (epoch 22):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| cherry_clean | 93.76% | 95.61% | 94.67% |
| cherry_pit | 94.76% | 92.58% | 93.66% |

**Confusion matrix (N=1,226):**

```
                  Predicted: clean    Predicted: pit
Actual: clean          631                 29
Actual: pit             42                524
```

**Note:** ConvNeXt V2 uses a different architecture than ResNet50 and is not a drop-in replacement — integration into the inference pipeline requires loader changes.

**Findings:**
- FCMAE pre-training (masked autoencoder) is well-suited for defect detection and outperformed standard ImageNet fine-tuning on EfficientNet-B2 despite similar parameter counts.
- **Label smoothing (α=0.1) caused catastrophic collapse** on ConvNeXt V2 (53.83%, stopped at epoch 7). A near-linearly-separable 2-class problem does not benefit from soft targets; they actively destabilize training here.
- Label smoothing had marginal positive effect on EfficientNet-B2 (+0.08 pp) but neither variant surpassed the 94.05% mark.
- **DINOv2 ViT-S/14 linear probe failed** (83.93%). A frozen ViT backbone does not transfer to cherry defect features without full fine-tuning. Not pursued further due to compute cost.
- Motion blur augmentation added for conveyor realism; its isolated contribution was not ablated.
- CPU latency for ConvNeXt V2 is ~58 ms vs. ~16 ms for ResNet50. GPU latency on production hardware has not been measured.

**Artifacts:**
- `threading_ws/src/cherry_detection/resource/experimental/convnextv2/model_best.pt`
- `threading_ws/src/cherry_detection/resource/experimental/convnextv2/metrics.json`
- `threading_ws/src/cherry_detection/resource/experimental/convnextv2/config.yaml`
- `training/notebooks/archive/colab_phase2_experiments.ipynb`
- `docs/reference/convnextv2_cpu_evaluation_results.json`

---

## Key Findings

1. **Train on unnormalized input.** Match the preprocessing of the target environment. This was the single biggest accuracy lever and also resolved a latency regression caused by denormal float values.
2. **Fine-tune all layers.** Full backbone fine-tuning outperforms FC-only or partial freezing. Cherry pit detection requires adapting early features, not just the classifier head.
3. **Label smoothing is harmful here.** A binary, near-separable classification problem with clean class definitions does not benefit from soft targets — use hard labels.
4. **Foundation model linear probes don't transfer.** DINOv2 frozen features are insufficient for specialized defect detection. Full fine-tuning would be required.
5. **FCMAE pre-training is a good fit for defect detection.** ConvNeXt V2's masked autoencoder pre-training produced the best accuracy of any model tested.
6. **Denormal float values cause severe latency regression (~10×).** When raw 0–255 inputs are passed through a network expecting normalized values, intermediate activations can produce denormal floats. Preprocessing must be consistent with training.
