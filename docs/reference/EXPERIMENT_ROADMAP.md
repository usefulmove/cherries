# Experimental Roadmap: Cherry Pit Classification Optimization

**Date:** 2026-02-06  
**Status:** Research Phase (pending)  
**Priority:** Address 3-class mismatch, then explore SOTA architectures  

---

## Executive Summary

This roadmap outlines a research-first approach to improving the cherry pit classification model. The current best model (ResNet50, 94.05% accuracy) achieves excellent performance but uses **2-class training** while production requires **3-class output** (clean/pit/maybe). Additionally, we will explore state-of-the-art architectures to potentially exceed current performance.

### Current Baseline

| Model | Accuracy | Latency | Size | Class Support | Status |
|-------|----------|---------|------|---------------|--------|
| **Best Training** (ResNet50) | 94.05% | ~17ms | 90MB | 2-class | Ready but mismatch |
| **Production** (ResNet50) | 92.99% | ~16ms | 90MB | 3-class (2-stage hack) | Currently deployed |
| **ResNet18 Alternative** | 91.92% | ~8-10ms | 43MB | 2-class | Speed-focused |

### Constraints
- **Latency Budget:** <30ms for classification stage (currently 16ms on ResNet50)
- **Inference Hardware:** CPU only (production constraint)
- **Training Infrastructure:** Google Colab Pro (Tesla T4 GPU)
- **Training Time:** ~30s/epoch, 30 epochs standard (~15-20 min per experiment)
- **Critical Gap:** No labeled "maybe" training data available

---

## Phase 1: SOTA Research & Brainstorming (Week 1-2)

### Objective
Before running experiments, deeply research what works for similar problems. Document findings and create evidence-based experimental proposals.

### Research Topics

#### 1.1 Three-Class Classification Strategies

**Problem:** We need 3-class output but only have 2-class training data (clean/pit). Production currently uses a problematic 2-stage training methodology.

**Research Questions:**
- How do industrial inspection systems handle "uncertain" categories without explicit training data?
- What are modern approaches for ambiguous class boundaries?
- Can we use post-training threshold calibration to create a "maybe" class?

**Approaches to Investigate:**
- **Option A: Post-hoc Threshold Calibration**
  - Train 2-class model, derive "maybe" from prediction confidence
  - Use probability thresholds (e.g., 0.5 < p(pit) < 0.75 = maybe)
  - Requires threshold optimization study
  
- **Option B: Synthetic "Maybe" Data Generation**
  - Generate ambiguous examples through data augmentation
  - Use Mixup/CutMix between clean and pit samples
  - Create synthetic edge cases

- **Option C: Soft Labeling with Uncertainty**
  - Use label smoothing on clean/pit classes
  - Train with probabilistic labels that allow uncertainty
  - Post-process into discrete classes

- **Option D: Evidential Deep Learning**
  - Model uncertainty explicitly in the architecture
  - Use evidential layers instead of softmax
  - Naturally produces "uncertain" predictions

**Deliverable:** Document comparing these 4 approaches with pros/cons for our specific case.

#### 1.2 Architecture Research (State-of-the-Art)

**Question:** What architectures from 2023-2025 could improve upon ResNet50 for fine-grained visual inspection?

**Research Areas:**

**A. Vision Transformers (ViTs) for Industrial Inspection**
- MobileViT (lightweight, mobile-optimized)
- EfficientFormer (efficient transformer variant)
- EdgeViT (edge-device optimized)
- **Question:** Can ViTs match CNNs for fine-grained defect detection with limited data?

**B. Modern CNN Architectures**
- ConvNeXt v2 (2023) - modernized CNN that competes with transformers
- EfficientNetV2 (2021, but still SOTA for efficiency)
- RepVGG (reparameterization for faster inference)
- **Question:** Do these actually outperform ResNet50 on small datasets?

**C. Hybrid CNN-Transformer Approaches**
- MobileFormer (parallel CNN + Transformer branches)
- EdgeNeXt (efficient hybrid)
- **Question:** Best of both worlds or unnecessary complexity?

**D. Attention Mechanisms**
- CBAM (Convolutional Block Attention Module)
- SE-Net (Squeeze-and-Excitation)
- **Question:** Can we add attention to existing ResNet50 for quick wins?

**Research Deliverables:**
1. List of 5-8 promising architectures with:
   - Parameter count (must be <50M for our latency budget)
   - Expected inference time on CPU
   - Evidence of performance on fine-grained classification tasks
   - Implementation availability (PyTorch pretrained weights?)

2. Preliminary rankings based on:
   - Potential accuracy improvement
   - Inference speed
   - Training stability with limited data
   - Implementation effort

#### 1.3 Training Methodology Innovations

**Research Questions:**
- Can we use self-supervised pre-training (SimCLR, MAE) to improve performance with limited labeled data?
- Would test-time augmentation (TTA) help with uncertain predictions?
- Can knowledge distillation from a larger teacher model (ResNet101, EfficientNet-B4) help?

**Approaches to Investigate:**
- **Self-Supervised Pre-training**
  - Use unlabeled cherry images for contrastive learning
  - Fine-tune on labeled clean/pit data
  - **Question:** Do we have enough unlabeled data? Is the effort worth it?

- **Test-Time Augmentation (TTA)**
  - Run inference on multiple augmented versions of same image
  - Average predictions for better calibration
  - **Question:** Can we afford multiple forward passes? (~48ms for 3x TTA)

- **Knowledge Distillation**
  - Train large teacher model (ResNet101, EfficientNet-B4)
  - Distill knowledge to smaller student (ResNet50, ResNet18)
  - **Question:** Can we achieve >94% accuracy with this approach?

#### 1.4 Data-Centric AI Strategies

**Research Questions:**
- Can we use active learning to identify which "maybe" examples to label?
- Can synthetic data generation (GANs/diffusion) help with edge cases?
- What about hard negative mining?

**Approaches:**
- **Active Learning**
  - Identify samples with high prediction uncertainty
  - Prioritize for human labeling
  - **Question:** Practical for production system?

- **Synthetic Data Generation**
  - Use diffusion models to generate edge cases
  - Create synthetic "maybe" examples
  - **Question:** Do we have the infrastructure? Is quality sufficient?

### Phase 1 Deliverables

1. **SOTA_RESEARCH.md** (in `docs/reference/`)
   - Section 1: 3-class strategies comparison
   - Section 2: Architecture research findings
   - Section 3: Training methodology innovations
   - Section 4: Data-centric strategies
   - Section 5: Prioritized recommendations (top 3-5 approaches)

2. **Architecture Shortlist**
   - 3-5 most promising architectures with justification
   - Implementation notes and pretrained weight availability

3. **Experimental Design Proposals**
   - 3-5 concrete experiments based on research findings
   - Each with: hypothesis, expected outcome, implementation plan

---

## Phase 2: Design & Prioritization (Week 2)

### Objective
Synthesize research findings into actionable experimental designs.

### Activities

1. **Architecture Selection**
   - Choose top 3 architectures to test
   - Consider: accuracy potential, speed, implementation effort, data requirements

2. **3-Class Strategy Decision**
   - Select primary approach for handling "maybe" class
   - Define threshold optimization methodology

3. **Experiment Design**
   - Define 3-5 specific experiments
   - Each experiment includes:
     - Clear hypothesis
     - Success metrics (accuracy, latency, calibration)
     - Implementation steps
     - Estimated training time
     - Risk assessment

4. **Baseline Definition**
   - Establish 94.05% ResNet50 (2-class) as baseline
   - Determine if we can measure 3-class performance without "maybe" labels

### Phase 2 Deliverables

1. **EXPERIMENT_SPECIFICATIONS.md**
   - Detailed spec for each proposed experiment
   - Clear pass/fail criteria
   - Resource requirements

2. **Prioritized Experiment List**
   - Ranked by: potential impact, implementation effort, risk
   - Dependencies identified

3. **Data Requirements Document**
   - What data we need for each experiment
   - Gaps and how to address them

---

## Phase 3: Implementation & Experiments (Weeks 3-6)

### Objective
Execute experiments in priority order.

### Experiment Candidates (to be refined after Phase 1)

**Candidate 1: Proper 3-Class Training with Synthetic "Maybe" Data**
- Use Mixup/CutMix to generate ambiguous samples between clean and pit
- Label these as "maybe" class
- Train true 3-class model
- Compare to 2-stage production approach
- **Risk:** Synthetic data may not represent real ambiguity

**Candidate 2: Modern Architecture Shootout**
- Test 3 top architectures from Phase 1 research
- ConvNeXt v2, EfficientNetV2, MobileViT
- Same training procedure (augmentation, unnormalized)
- Measure accuracy vs latency tradeoff
- **Risk:** May not outperform ResNet50 on small dataset

**Candidate 3: Threshold Optimization Study**
- Systematic threshold search on best 2-class model
- Optimize for pit recall vs precision
- Create confidence-calibrated "maybe" class
- **Risk:** May not significantly improve over current thresholds

**Candidate 4: Knowledge Distillation**
- Train large teacher (ResNet101 or EfficientNet-B4) on Colab
- Distill to student (ResNet50 or ResNet18)
- Compare student accuracy to baseline
- **Risk:** May require more training time/compute

**Candidate 5: Attention Mechanism Enhancement**
- Add CBAM or SE blocks to ResNet50
- Quick architecture modification
- Test if attention improves accuracy
- **Risk:** May increase latency beyond budget

### Implementation Protocol

1. **Colab Notebook Preparation**
   - Create dedicated notebook per experiment
   - Include smoke test mode (1 epoch, CPU)
   - GPU verification at start
   - Automatic metrics logging

2. **Training Procedure**
   - Use established augmentation pipeline (rotation, affine)
   - Unnormalized training (0-255 range to match production)
   - 30 epochs, early stopping if no improvement for 10 epochs
   - Save all checkpoints and metrics

3. **Evaluation**
   - Primary metric: accuracy on validation set
   - Secondary: per-class precision/recall
   - Latency benchmark on CPU
   - Calibration quality (confidence vs accuracy)

### Phase 3 Deliverables

1. **Trained Model Weights** (in `training/experiments/`)
2. **Metrics JSON** for each experiment
3. **Latency Benchmarks** for CPU inference
4. **Comparison Report** vs 94.05% baseline

---

## Phase 4: Evaluation & Decision (Week 7)

### Objective
Analyze results and make deployment recommendations.

### Activities

1. **Performance Analysis**
   - Statistical comparison of all models
   - Significance testing for accuracy differences
   - Latency vs accuracy tradeoff curves

2. **3-Class Performance Assessment**
   - How well does each model handle "maybe" class?
   - Compare against production 2-stage approach
   - Measure pit recall (critical for food safety)

3. **Deployment Readiness Check**
   - Model size acceptable? (<100MB)
   - Latency within budget? (<30ms)
   - No denormal values (latency killer)
   - Compatible with existing preprocessing

4. **Recommendation Document**
   - Which model(s) to deploy?
   - What deployment process?
   - Risk assessment
   - Rollback plan

### Phase 4 Deliverables

1. **FINAL_EVALUATION.md** with:
   - All results comparison table
   - Statistical analysis
   - Deployment recommendation

2. **Deployment Package**:
   - Selected model weights
   - Updated inference code (if needed)
   - Configuration parameters

3. **Updated Documentation**:
   - Model card for deployed model
   - LESSONS.md updates
   - OPERATIONS.md updates

---

## Success Criteria

### Minimum Viable Success
- **Deliver:** Comprehensive SOTA research document
- **Deliver:** 3-5 concrete experimental proposals
- **Deliver:** Decision on 3-class strategy

### Target Success
- **Achieve:** At least one model with >94.05% accuracy (2-class) OR
- **Achieve:** 3-class model with >92% accuracy on all classes
- **Maintain:** Latency <30ms on CPU
- **Demonstrate:** Clear improvement over production baseline

### Stretch Goal
- **Achieve:** >95% accuracy OR
- **Achieve:** ResNet18-level speed (<10ms) with ResNet50-level accuracy (94%)
- **Innovate:** Novel approach to 3-class without labeled "maybe" data

---

## Resources & References

### Key Documents
- `docs/reference/MODEL_EXPERIMENTS.md` - Previous experiment results
- `docs/reference/TRAINING_METHODOLOGY.md` - Current 2-stage approach critique
- `training/notebooks/colab_optimization_experiments.ipynb` - Working training pipeline
- `docs/reference/architecture-quick-reference.md` - System constraints

### Infrastructure
- **Training:** Google Colab Pro (Tesla T4 GPU)
- **Testing:** Local CPU for inference benchmarking
- **Data:** `training/data/` (clean/pit images)
- **Code:** `training/scripts/train.py` (training library)

### Budget
- **Training Time:** ~15-20 minutes per experiment (30 epochs)
- **Compute:** Colab Pro GPU hours (estimate 5-10 hours total)
- **Human Time:** 4-6 weeks of research + experimentation

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No "maybe" data limits 3-class approaches | High | High | Focus on synthetic generation and threshold optimization |
| Modern architectures don't outperform ResNet50 | Medium | Medium | Research thoroughly before implementation; have fallback plan |
| Training runs on CPU (no GPU) | Low | High | Add hard GPU check at start of all notebooks |
| Colab Pro GPU hours exhausted | Low | Medium | Batch experiments efficiently; use smoke tests first |
| Models exceed latency budget | Medium | Medium | Benchmark latency early; prioritize speed-oriented architectures |
| No significant accuracy improvement | Medium | Medium | Document learnings; recommend threshold optimization instead |

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1: Research | Week 1-2 | SOTA_RESEARCH.md with prioritized recommendations |
| Phase 2: Design | Week 2 | EXPERIMENT_SPECIFICATIONS.md |
| Phase 3: Experiments | Week 3-6 | Trained models + comparison report |
| Phase 4: Decision | Week 7 | FINAL_EVALUATION.md + deployment recommendation |

**Total Duration:** 7 weeks (can be parallelized if multiple Colab sessions available)

---

## Next Steps (Immediate Actions)

1. **Begin Phase 1 Research** (today)
   - Start with 3-class classification strategies
   - Survey recent papers on ambiguous class boundaries
   - Check if ConvNeXt/EfficientNetV2 have pretrained PyTorch weights

2. **Data Inventory** (this week)
   - Confirm exact number of clean/pit training samples
   - Check if any unlabeled cherry images exist for self-supervised pre-training
   - Document any data augmentation parameters from previous experiments

3. **Baseline Verification** (this week)
   - Confirm 94.05% ResNet50 model is accessible
   - Re-run latency benchmark to verify ~17ms on CPU
   - Document production threshold logic for "maybe" class

---

**Document Owner:** dedmonds  
**Last Updated:** 2026-02-06  
**Status:** Ready for Phase 1 execution
