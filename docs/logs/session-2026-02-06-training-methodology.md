# Session: Two-Stage Training Methodology Assessment

**Date:** 2026-02-06
**Topic:** Analysis and documentation of cherry classification two-stage training approach
**Status:** Complete

---

## Session Summary

Discovered and documented that the production cherry classification model was trained using an unconventional two-stage methodology with significant safety and architectural concerns.

---

## What Was Accomplished

### 1. **Primary Documentation Created**

**File:** `docs/reference/TRAINING_METHODOLOGY.md`

Comprehensive documentation covering:
- Two-stage training process (binary → 3-class via error fine-tuning)
- Five critical concerns identified through expert analysis:
  1. Food safety risk (Stage 1 misses become permanent)
  2. Architecture violation ("maybe" class defined by errors, not ground truth)
  3. Catastrophic forgetting (Stage 2 erodes Stage 1 learning)
  4. Production complexity (cascade validation burden)
  5. Threshold redundancy (3-class output converted back to thresholds anyway)
- Three recommended alternatives ranked by preference
- Proposed comparison experiment design

### 2. **Architecture Documentation Updated**

**File:** `docs/core/architecture/inference_pipeline/ARCHITECTURE.md`

Added:
- Training Methodology Assessment section with summary table
- Key concerns list
- Recommended alternatives
- Cross-references to methodology document
- Technical debt entry noting methodology concerns

### 3. **Experiment Log Updated**

**File:** `docs/reference/MODEL_EXPERIMENTS.md`

Added Important Context section explaining:
- All experiments used 2-class classification (clean/pit only)
- Production uses 3-class via two-stage methodology
- Creates critical discrepancy: best experimental model (94.05%) doesn't match production architecture
- Two-stage approach has safety risks

### 4. **Analysis Document Updated**

**File:** `docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md`

Added context to threshold section:
- Analysis shows 2-class architecture but production uses 3-class
- "Maybe" class created synthetically from Stage 1 misclassifications
- References to methodology documentation

### 5. **Training Data Reference Updated**

**File:** `docs/reference/training-data.md`

Updated Classification Model section:
- Clarified training data is inherently 2-class (clean/pit)
- "Maybe" class synthetically created, not manually labeled
- Added methodology cross-reference

### 6. **Lessons Learned Updated**

**File:** `docs/reference/LESSONS.md`

Added lesson entry [2026-02-06]:
- Concise summary of two-stage methodology discovery
- Key assessment points
- Recommended alternatives
- Cross-references to related docs

---

## Key Findings

### The Two-Stage Approach

```
Stage 1: Train binary classifier (pit vs no_pit)
         ↓
    Identify misclassifications ("misses")
         ↓
Stage 2: Fine-tune on misses as "maybe" class
         ↓
    3-class output (clean, maybe, pit)
```

### Critical Issues

1. **Safety Risk:** Stage 1 misses become "clean" forever—pits classified as clean in Stage 1 cannot be recovered in Stage 2
2. **Architecture Flaw:** "Maybe" class violates cross-entropy's mutual exclusivity assumption; defined by model errors rather than ground truth
3. **Learning Destruction:** Stage 2 fine-tuning without replay or proper LR management erodes Stage 1 decision boundary
4. **Operational Burden:** Two models must version-lock; impossible to rollback Stage 1 alone
5. **Redundancy:** Production converts 3-class output back to thresholds (0.5/0.75) anyway

### Recommended Alternatives

1. **Current 3-Class Explicit** (status quo): Manually label "maybe" examples, train from scratch
2. **Enhanced 2-Class**: Binary classifier with calibrated confidence tiers (pit≥0.85 auto-reject, 0.65-0.85 review, <0.65 accept)  
3. **Ensemble Methods**: Multiple models; disagreement indicates uncertainty

---

## Discrepancy Identified

**Critical:** All recent optimization experiments (including the 94.05% accuracy best model) used **2-class classification**, while production uses **3-class** via the problematic two-stage approach.

**Impact:** The best experimental model cannot be directly deployed to production without architectural changes or methodology alignment.

---

## Files Modified

1. `docs/reference/TRAINING_METHODOLOGY.md` (created)
2. `docs/core/architecture/inference_pipeline/ARCHITECTURE.md` (updated)
3. `docs/reference/MODEL_EXPERIMENTS.md` (updated)
4. `docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md` (updated)
5. `docs/reference/training-data.md` (updated)
6. `docs/reference/LESSONS.md` (updated)

---

## Next Steps

1. **Verify production model training details** - Confirm exact process used for `classification-2_26_2025-iter5.pt`
2. **Design comparison experiment** - Test conventional 3-class vs two-stage approach
3. **Locate original training notebooks** - Understand the actual process and rationale
4. **Evaluate deployment path** - Decide whether to:
   - Keep current 3-class model (accept known issues)
   - Retrain with proper methodology
   - Simplify to enhanced 2-class approach

---

## References

- Primary: [Training Methodology](../reference/TRAINING_METHODOLOGY.md)
- Architecture: [Inference Pipeline](../core/architecture/inference_pipeline/ARCHITECTURE.md)
- Experiments: [Model Experiments](../reference/MODEL_EXPERIMENTS.md)
- Lessons: [LESSONS.md](../reference/LESSONS.md) entry [2026-02-06]
