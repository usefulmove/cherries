# Session: Phase 1 Research Complete

**Date:** 2026-02-06  
**Phase:** Phase 1 - State-of-the-Art Research (COMPLETE)  
**Status:** Ready for Phase 2 - Design & Preparation

---

## Session Overview

Completed comprehensive research on SOTA architectures, 3-class classification strategies, and attention mechanisms for cherry pit detection optimization. Research is now sound and ready for Phase 3 experiment design.

---

## Research Completed

### 1. Architecture Research ✓

**Top Candidates Identified:**

| Architecture | Params | Est. Latency | ImageNet Top-1 | Status |
|--------------|--------|--------------|----------------|--------|
| ConvNeXt-Tiny | 28M | ~17ms | 82.1% | Primary candidate |
| EfficientNet-B2 | 9M | ~12ms | 80.1% | Test both norm variants |
| ECA-ResNet50 | 26M | ~17ms | 77.4% (+1.3%) | Quick win |
| SE-ResNet50 | 28M | ~18ms | 77.6% (+1.5%) | Alternative |

### 2. Attention Mechanism Research ✓

**Recommendation: ECA (Efficient Channel Attention)**
- Adds only 0.15M parameters (0.5% overhead)
- Latency impact: +0.5-1ms
- Accuracy gain: +1.3% on ImageNet
- Best for small datasets, simple implementation
- Available in timm: `ecaresnet50`

**Alternatives:**
- SE: +2.5M params, +1.5% accuracy, +1-2ms latency
- CBAM: +4-5M params, +1.8% accuracy, +3-5ms latency

### 3. PyTorch Model Availability ✓

All models verified available:
- **ConvNeXt-Tiny**: `torchvision.models.convnext_tiny` ✓
- **EfficientNet-B2**: `torchvision.models.efficientnet_b2` ✓
- **ECA-ResNet50**: `timm.create_model('ecaresnet50')` ✓
- **SE-ResNet50**: `timm.create_model('seresnet50')` ✓

**Critical Note:** All pretrained models expect normalized (0-1) input, but production uses unnormalized (0-255).

### 4. Knowledge Distillation Assessment ✓

**Verdict: MEDIUM Priority**
- Teacher: Existing 94.05% ResNet50 (no training cost)
- Student: ResNet18 with distillation
- Expected: ~93% accuracy (up from 91.92% direct training)
- Implementation: ~30 lines of code
- Unlikely to beat 94.05% baseline, but could make ResNet18 viable

### 5. Data & Augmentation Analysis ✓

**Current State:**
- ~5k total samples (train/val split)
- 2-class only (clean/pit) - no "maybe" training data
- Good augmentation: flips, rotation (180°), affine, color jitter
- Recent 2024 production data (11k+ images) not yet integrated

**Augmentation Gaps:**
- No Mixup/CutMix (for synthetic "maybe" data)
- No label smoothing (alpha=0.1 recommended)
- No AutoAugment

### 6. Three-Class Strategy ✓

**Recommendation: Threshold Calibration (Option A)**

Ranked strategies:
1. **Threshold Calibration** (LOW complexity, HIGH recommendation)
2. **Synthetic Data (Mixup)** (MEDIUM complexity, MEDIUM recommendation)
3. **Label Smoothing** (LOW complexity, MEDIUM recommendation)
4. **Evidential Deep Learning** (HIGH complexity, LOW recommendation)

**Critical Blocker:** No labeled "maybe" training data available.

### 7. Normalization Decision ✓

**User Decision: Option B**
- Primary approach: Unnormalized (0-255) - proven superior (94.05% vs 93.96%)
- **EfficientNet-B2 only: Test BOTH normalized AND unnormalized**

**Evidence from MODEL_EXPERIMENTS.md:**
- ResNet50 Augmented (Normalized): 93.96%, ~37ms
- **ResNet50 Augmented (Unnormalized): 94.05%, ~16.7ms ← Winner**

Rationale: Drop-in replacement advantage + slightly better accuracy + dramatically better latency.

---

## Final Architecture Decision Matrix

| Rank | Architecture | Params | Est. Latency | Accuracy Potential | Normalization | Verdict |
|------|--------------|--------|--------------|-------------------|---------------|---------|
| 1 | **ConvNeXt-Tiny** | 28M | ~17ms | 94.5-95% | Unnormalized | **PRIMARY** |
| 2 | **EfficientNet-B2** | 9M | ~12ms | 93.5-94.5% | **BOTH** | Test both (per user) |
| 3 | **ECA-ResNet50** | 26M | ~17ms | 94.3-94.8% | Unnormalized | Quick win |
| 4 | **ResNet18 + Distillation** | 12M | ~9ms | ~93% | Unnormalized | Speed/size play |
| 5 | ConvNeXt-Small | 50M | ~25ms | 95%+ | Unnormalized | At limit |
| 6 | MobileViT-S | 6M | ~11ms | 92-93% | Unnormalized | Uncertain |

---

## Phase 3 Experiments (Prioritized)

### HIGH Priority

**Exp 1: Threshold Optimization**
- Goal: Optimize "maybe" class thresholds on 94.05% model
- Duration: 1 day
- Normalization: Unnormalized (existing model)
- Success: Define production-ready thresholds

**Exp 2: ConvNeXt-Tiny**
- Goal: Test modernized architecture vs ResNet50
- Duration: 1 day
- Normalization: Unnormalized
- Success: ≥94% accuracy

### MEDIUM Priority

**Exp 3: EfficientNet-B2 (Dual Test)**
- Goal: Verify if previous B0 failure was normalization-related
- Duration: 2 days (both variants)
- Normalization: **Both normalized AND unnormalized**
- Success: Either variant >93.5%

**Exp 4: ECA-ResNet50**
- Goal: Quick accuracy boost with attention
- Duration: 1 day
- Normalization: Unnormalized
- Success: >94.3% accuracy

### LOW-MEDIUM Priority

**Exp 5: Knowledge Distillation**
- Goal: Make ResNet18 viable via distillation
- Duration: 4 hours + training
- Normalization: Unnormalized
- Success: ≥93.5% accuracy

---

## Critical Constraints

1. **No "maybe" training data** - must use threshold calibration
2. **All models use unnormalized training** - proven superior
3. **30ms latency budget** - all candidates well within limit
4. **Colab Pro (Tesla T4)** - sufficient for all experiments
5. **Model size limit: 50M params** - ConvNeXt-Small at boundary

---

## Baseline Models

| Model | Accuracy | Latency | Params | Size | Class Support | Location |
|-------|----------|---------|--------|------|---------------|----------|
| **Best Training** | 94.05% | ~17ms | 25.6M | 90MB | 2-class | `training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt` |
| **Production** | 92.99% | ~16ms | 25.6M | 90MB | 3-class (2-stage) | `cherry_system/cherry_detection/resource/cherry_classification.pt` |
| **ResNet18 Alt** | 91.92% | ~8-10ms | 11.7M | 43MB | 2-class | `training/experiments/resnet18_augmented_unnormalized/model_best.pt` |

---

## Code Infrastructure Status

**Existing & Ready:**
- ✓ Threshold optimization script: `training/scripts/optimize_thresholds.py`
- ✓ Training infrastructure: `training/scripts/train.py`
- ✓ Colab notebook: `training/notebooks/colab_optimization_experiments.ipynb`
- ✓ Model support: ResNet18/50 in `training/src/model.py`

**Needs Implementation (Phase 2):**
- ☐ Add ConvNeXt support to `model.py`
- ☐ Add EfficientNet-B2 support to `model.py`
- ☐ Add ECA-ResNet50 support (via timm)
- ☐ Add distillation loss function option to `train.py`
- ☐ Create experiment configs for all 5 experiments
- ☐ Add normalized variant support for EfficientNet-B2

---

## Next Session: Phase 2 - Design & Preparation

**Focus Areas:**
1. Update `training/src/model.py` with new architectures
2. Add distillation loss to training loop
3. Create experiment configs
4. Decide: Start with threshold optimization (no code changes) OR model.py updates first?

**Reference Files:**
- `docs/reference/EXPERIMENT_ROADMAP.md` - Complete roadmap
- `docs/reference/SOTA_RESEARCH.md` - Research findings
- `docs/reference/MODEL_EXPERIMENTS.md` - Previous experiment results
- This session log

---

**Session saved and ready for Phase 2 implementation.**
