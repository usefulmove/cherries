# Cherry Classification: Training Methodology

## Overview

The cherry pit classifier was trained using a **two-stage approach** that first performs binary classification, then fine-tunes on misclassified examples to create a third "uncertain" class. This document explains the methodology, documents the known process, and outlines concerns with this approach.

---

## Current Production Model

**Model:** `classification-2_26_2025-iter5.pt`  
**Location:** `threading_ws/src/cherry_detection/resource/`  
**Architecture:** ResNet50 (25.6M parameters)  
**Classes:** 3-class (clean, maybe, pit)  
**Input:** 128×128 pixel crops, unnormalized (0-255 range)  
**Accuracy:** ~92-94% (validation set)

---

## Two-Stage Training Methodology

### Stage 1: Binary Classification

**Objective:** Train a pit vs. no-pit classifier

**Process:**
- Dataset: Clean cherries (no pit) vs. cherries with pits
- Classes: 2 (pit, no_pit)
- Training: Standard fine-tuning of ResNet50
- Output: Binary classifier with softmax over 2 classes

**Dataset Source:**
- Primary: `https://github.com/weshavener/cherry_classification`
- Collected: November 2, 2022
- Distribution: ~54% clean, 46% pit (well-balanced)

### Stage 2: Fine-Tuning on "Misses"

**Objective:** Create a third class for uncertain predictions

**Process:**
1. Run Stage 1 model on validation/training data
2. Identify misclassified examples ("misses")
3. Label these misses as "maybe" class
4. Fine-tune the model on 3 classes: clean, maybe, pit

**Key Characteristics:**
- The "maybe" class consists entirely of examples the Stage 1 model got wrong
- This includes both false positives and false negatives
- The model learns to recognize its own failure modes

---

## Historical Evolution

| Version | Approach | Classes | Model File | Notes |
|---------|----------|---------|------------|-------|
| v1-v5 | 2-class + thresholds | 2 (clean/pit) | `cherry_classification.pt` | "Maybe" via threshold (0.5-0.75) |
| v6+ | 3-class explicit | 3 (clean/maybe/pit) | `classification-2_26_2025-iter5.pt` | Current production |

**Key Change:** The transition from threshold-based "maybe" (0.5 ≤ pit_prob < 0.75) to explicit 3-class prediction (clean=0, maybe=1, pit=2).

---

## Concerns with Two-Stage Approach

### 1. **Systematic Bias Risk**

The "maybe" class is not a natural category—it's defined entirely by model errors. This creates several risks:
- **Error accumulation:** Stage 2 learns from Stage 1's mistakes, potentially inheriting and amplifying biases
- **Distribution shift:** The "maybe" samples may not represent true uncertainty but rather systematic failure modes
- **Feedback loop:** If Stage 1 has blind spots, Stage 2 learns to recognize those blind spots rather than fix them

### 2. **Class Definition Ambiguity**

Unlike clean/pit which have objective definitions, "maybe" is defined negatively:
- Not: "Cherries that are genuinely ambiguous"
- But: "Cherries the first model couldn't classify correctly"

This conflates:
- Truly ambiguous cherries (e.g., unusual lighting, partial occlusion)
- Systematic errors (e.g., specific orientations, pit types)
- Labeling errors (mislabeled training data)

### 3. **Learning Preservation Uncertainty**

Unknown: Does Stage 2 preserve Stage 1 learning or erode it?
- Fine-tuning on 3 classes may overwrite Stage 1's binary decision boundary
- The "maybe" class could dilute the clean/pit distinction
- No validation that Stage 2 maintains Stage 1 performance on original binary task

### 4. **Threshold Dependency**

Production system still uses thresholds after 3-class classification:
- pit_prob ≥ 0.75 → label 2 (pit)
- maybe detected → label 5 (uncertain)
- clean_prob ≥ 0.5 → label 1 (clean)

This suggests the 3-class output is being converted back to threshold-based decisions, raising questions about the value of the two-stage training.

---

## Expert Assessment Summary

**Verdict:** The two-stage approach has critical flaws and is not recommended for production use.

| Concern | Severity | Impact |
|---------|----------|--------|
| **Food Safety Risk** | Critical | Stage 1 misses become permanent false negatives; pits classified as "clean" cannot be recovered |
| **Architecture Violation** | High | "Maybe" class defined by errors, not ground truth; violates mutual exclusivity assumption in cross-entropy loss |
| **Catastrophic Forgetting** | High | Stage 2 fine-tuning erodes Stage 1 learning without replay or proper learning rate management |
| **Production Complexity** | Medium | Two models must version-lock; impossible to rollback Stage 1 alone; creates cascade validation burden |
| **Threshold Redundancy** | Medium | 3-class output converted back to thresholds anyway; unnecessary indirection |
| **Auditability** | Medium | Training data contaminated by Stage 1 errors; hard to justify to regulators |

**Recommended Alternatives (in order of preference):**

1. **Current 3-Class Explicit** (status quo): Keep production model but note concerns; manually label "maybe" examples for future retraining
2. **Enhanced 2-Class with Confidence Tiers**: Binary classifier with calibrated thresholds (pit≥0.85 auto-reject, 0.65-0.85 review, <0.65 accept)
3. **Ensemble Methods**: Multiple models; prediction disagreement indicates uncertainty

See LESSONS.md entry [2026-02-06] for concise takeaway.

---

## Alternative Approaches

### Approach A: Conventional 3-Class Training

**Process:** Train a single 3-class model from scratch with three explicitly labeled classes.

**Advantages:**
- Clean, well-defined classes from the start
- No risk of error propagation between stages
- Simpler training pipeline
- Easier to interpret and debug

**Challenges:**
- Requires manual labeling of "maybe" examples
- Need clear criteria for what constitutes "uncertain"
- May need more training data for the minority "maybe" class

### Approach B: Curriculum Learning

**Process:** Train on easy examples first (clear clean/pit), then gradually introduce harder examples.

**Advantages:**
- Natural progression from simple to complex
- "Hard" examples become the "maybe" class organically
- Preserves strong baseline performance on easy cases

**Challenges:**
- Requires difficulty scoring of training examples
- More complex training schedule

### Approach C: Confidence-Based Routing

**Process:** Train binary classifier, use confidence threshold to route uncertain predictions to manual review.

**Advantages:**
- Simple, interpretable system
- No need for "maybe" class at all
- Direct control over uncertainty handling

**Challenges:**
- Requires human-in-the-loop for uncertain cases
- May have high false positive rate on uncertain predictions

---

## Proposed Comparison Experiment

To evaluate whether the two-stage approach is beneficial, we propose a controlled comparison:

### Experiment Design

| Aspect | Current (Two-Stage) | Conventional (3-Class) |
|--------|---------------------|------------------------|
| **Architecture** | ResNet50 | ResNet50 |
| **Classes** | 3 (via 2-stage) | 3 (simultaneous) |
| **Training Data** | Same underlying dataset | Same underlying dataset |
| **Data Split** | Same train/val/test | Same train/val/test |
| **Preprocessing** | Unnormalized (0-255) | Unnormalized (0-255) |
| **Augmentation** | Same augmentation pipeline | Same augmentation pipeline |
| **Epochs** | 10 + 10 (stage 1 + stage 2) | 20 total |
| **Metrics** | Accuracy, per-class precision/recall | Same |

### Success Criteria

Measure which approach achieves:
1. Higher overall accuracy
2. Better pit recall (minimize false negatives)
3. More balanced precision across classes
4. Lower confusion between clean/pit (the critical error)

### Key Questions to Answer

1. Does the two-stage approach improve or degrade clean vs. pit separation?
2. Is the "maybe" class actually useful, or does it just dilute decisions?
3. Can we define "maybe" more explicitly (e.g., via manual labeling) to improve the conventional approach?
4. What is the business cost of false negatives (missed pits) vs. false positives (false alarms)?

---

## Known Unknowns

**Questions needing answers:**
1. Where are the original training notebooks/scripts for the two-stage process?
2. What were the specific hyperparameters for each stage?
3. How was the "maybe" class threshold defined in Stage 2?
4. Was there validation that Stage 2 preserved Stage 1 binary performance?
5. What motivated the shift from 2-class to 3-class in production?

**Hypotheses to test:**
1. The two-stage approach may have been driven by data availability (not enough explicitly labeled "maybe" examples)
2. Stage 2 may have been an attempt to "fix" Stage 1 errors rather than retrain from scratch
3. The production thresholds (0.5, 0.75) may be suboptimal for the 3-class output

---

## Recommendations

### Immediate

1. **Document training process** (this file) - ✅ In progress
2. **Locate original training notebooks** - Understand the exact process used
3. **Validate current model behavior** - Confirm production model matches documented approach

### Short-term

1. **Design comparison experiment** - Test conventional 3-class vs. two-stage
2. **Collect explicitly labeled "maybe" examples** - For conventional approach training
3. **Analyze production error patterns** - Understand what types of cherries end up as "maybe"

### Long-term

1. **Evaluate curriculum learning** - If data volume allows
2. **Consider confidence-based routing** - Simpler alternative to 3-class
3. **Automated retraining pipeline** - Continuous learning from production feedback

---

## Related Documentation

- [Model Experiments](./MODEL_EXPERIMENTS.md) - Recent optimization experiments
- [Optimization Findings](./optimization-findings-summary.md) - Current best practices
- [Inference Pipeline Architecture](../core/architecture/inference_pipeline/ARCHITECTURE.md) - How classification integrates into system
- [Training Data Reference](./training-data.md) - Dataset details
- [ResNet50 Analysis](../core/architecture/inference_pipeline/RESNET50_ANALYSIS.md) - Architecture deep-dive

---

## Status

**This document is a draft based on conversation and code analysis.** The two-stage training methodology was described by the user as discovered knowledge, not documented in existing codebase. This document captures the approach as understood and raises concerns for future investigation.

**Next Steps:**
- [ ] Locate original training notebooks/scripts
- [ ] Verify production model training details
- [ ] Design comparison experiment
- [ ] Update this document with experiment results
