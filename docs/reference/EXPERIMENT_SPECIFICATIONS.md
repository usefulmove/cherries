# Experiment Specifications: Phase 2 Optimization

**Date:** 2026-02-06  
**Phase:** Phase 2 - Design & Prioritization  
**Status:** Ready for Implementation  
**Baseline:** ResNet50 94.05% accuracy (2-class), 16ms latency

---

## Overview

This document specifies the exact experimental designs for our next optimization cycle. Based on Phase 1 research and external expert feedback (Claude, Gemini, GPT), we are pivoting to **state-of-the-art architectures** and **domain-specific augmentations**.

**Key Changes from Previous Plan:**
1.  **DINOv2 (Vision Transformer):** Added as top-tier experiment (linear probe).
2.  **ConvNeXt V2:** Upgraded from V1 to V2 (using `timm`) for better defect detection.
3.  **Enhanced Augmentations:** Added Motion Blur and Photometric Distortion to all experiments to match conveyor belt conditions.

**Selected Experiments:**
1.  **EXP-001:** Threshold optimization (immediate value, no training)
2.  **EXP-002:** ConvNeXt V2-Tiny (via `timm`, modernized architecture)
3.  **EXP-003:** EfficientNet-B2 (Unnormalized, speed-focused)
4.  **EXP-006:** DINOv2 Linear Probe (Foundation model, frozen backbone)
5.  **EXP-005:** Label Smoothing (A/B test across all)

**Execution Strategy:** Parallel (threshold optimization + 1 architecture simultaneously)

---

## Success Criteria Framework

### Primary Metrics (All Experiments)

| Metric | Definition | Target | Rationale |
|--------|-----------|--------|-----------|
| **Accuracy** | (TP + TN) / Total | ≥94.05% | Must beat or match baseline |
| **Pit Recall** | TP / (TP + FN) | ≥99.0% | Food safety critical - can't miss pits |
| **Clean Precision** | TN / (TN + FP) | ≥95.0% | Don't waste product on false alarms |
| **Latency** | CPU inference time | ~16ms | Production Baseline (faster on GPU) |
| **Model Size** | .pt file size | <100MB | Deployment constraint |

---

## Global Configuration: Enhanced Augmentation
**Applied to ALL Training Experiments (EXP-002, 003, 006)**

Research indicates existing augmentations miss "conveyor belt lies" (motion blur, lighting shifts).

**Action:** Update `src/data.py` before starting experiments.

```python
# Enhanced transforms.Compose list:
[
    # Geometric (Existing)
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    
    # NEW: Conveyor Realism (Critical)
    transforms.RandomMotionBlur(kernel_size=5, angle_range=(-90, 90), direction=0.5, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    
    # Base
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
]
```

---

## Experiment 1: Threshold Optimization
**ID:** EXP-001  
**Type:** Analysis (no training)  
**Priority:** HIGH  
**Objective:** Optimize decision boundaries for 3-class system on existing 94.05% model.  
*(Unchanged from previous revision - see full details in original doc if needed)*

---

## Experiment 2: ConvNeXt V2-Tiny (Modernized)
**ID:** EXP-002  
**Type:** Training  
**Priority:** HIGH  
**Depends On:** `pip install timm`

### Objective
Test **ConvNeXt V2** (not V1). V2 uses Masked Auto-Encoder (MAE) pre-training, making it superior for "filling in the blanks" (defect detection).

### Architecture Details
```python
# Source: timm (pytorch-image-models)
# Model: convnextv2_tiny.fcmae_ft_in1k
import timm
model = timm.create_model(
    'convnextv2_tiny.fcmae_ft_in1k',
    pretrained=True,
    num_classes=2
)
```
- **Params:** ~28M
- **Input:** 128x128 (Unnormalized 0-255 preferred if supported, else standard)

### Training Configuration
- **Optimizer:** AdamW (Required for ConvNeXt)
- **LR:** 1e-4
- **Augmentation:** Enhanced (Motion Blur enabled)
- **Label Smoothing:** 0.1 (Default for V2 training)

### Success Criteria
- Accuracy ≥ 94.5%
- Latency < 25ms

---

## Experiment 6: DINOv2 Linear Probe (Foundation Model)
**ID:** EXP-006  
**Type:** Training (Frozen Backbone)  
**Priority:** HIGH (Tier 1 Recommendation)

### Objective
Use Meta's DINOv2 (ViT-Small) as a feature extractor. DINOv2 excels at fine-grained part correspondence (e.g., finding a pit) even without fine-tuning.

### Architecture Details
```python
# Load from Torch Hub
# Model: dinov2_vits14 (Small, 14x14 patch)
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

# Add Linear Head
# ViT-S embedding dimension is 384
class DinoClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(384, 2)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

### Hypothesis
A frozen DINOv2 backbone with a simple linear head will outperform fully fine-tuned ResNet50 due to superior representation learning, and will train extremely fast (only ~1k params to optimize).

### Success Criteria
- Accuracy ≥ 94.5%
- Latency check (ViT can be slower; if slow but accurate, candidate for Distillation)

---

## Experiment 3: EfficientNet-B2 (Unnormalized)
**ID:** EXP-003  
**Type:** Training  
**Priority:** MEDIUM (Speed alternative)

### Objective
Retest EfficientNet-B2 with **unnormalized training** and **Enhanced Augmentations**. Previous B0 failure likely due to normalization mismatch.

### Configuration
- **Model:** `efficientnet_b2` (torchvision)
- **Input:** Unnormalized (0-255)
- **Augmentation:** Enhanced (Motion Blur)

---

## Execution Plan

### Phase 2A: Setup & Baselines (Day 1)
1.  **Install Requirements:** `pip install timm`
2.  **Update Code:**
    - Modify `src/data.py` (Add Motion Blur)
    - Modify `src/model.py` (Add `timm` and `dinov2` support)
3.  **EXP-001:** Run threshold optimization on current best model.

### Phase 2B: The "SOTA" Shootout (Day 2-3)
Run in parallel if possible:
1.  **EXP-002 (ConvNeXt V2):** The modern CNN champion.
2.  **EXP-006 (DINOv2):** The Foundation Model challenger.

**⚠️ DINOv2 Resolution Constraint:** DINOv2 requires input dimensions to be a multiple of the patch size (14x14). Changed `input_size` from **128** to **126** (14 * 9) in the config.

### Phase 2C: Speed Option (Day 3)
1.  **EXP-003 (EfficientNet-B2):** Only if EXP-002/006 are too slow (significantly worse than 16ms baseline).

### Decision Logic
| Outcome | Action |
|---------|--------|
| **DINOv2 Wins** (High Acc, OK Speed) | Deploy DINOv2 (Linear Probe). |
| **DINOv2 Wins** (High Acc, Slow Speed) | Use DINOv2 as **Teacher** for Distillation (Phase 3). |
| **ConvNeXt V2 Wins** | Deploy ConvNeXt V2. |
| **All Fail** | Fallback to ResNet50 + SE Blocks (EXP-004). |

---

## Implementation Checklist

- [ ] **Dependencies:** Install `timm` in Colab environment.
- [ ] **Data Pipeline:** Update `src/data.py` with `RandomMotionBlur` and stronger `ColorJitter`.
- [ ] **Model Factory:** Update `src/model.py`:
    - [ ] Add `create_convnext_v2` using `timm`.
    - [ ] Add `create_dinov2` using `torch.hub`.
- [ ] **Configs:** Create YAMLs for EXP-002 (ConvNeXt V2) and EXP-006 (DINOv2).
- [ ] **Notebook:** Update `colab_phase2_experiments.ipynb` to install dependencies.

---
**Document Owner:** dedmonds
**Last Updated:** 2026-02-06
