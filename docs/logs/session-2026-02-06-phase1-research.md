# Session: Phase 1 Research - SOTA Classification Architectures

**Date:** 2026-02-06  
**Duration:** ~45 minutes  
**Status:** Phase 1 research initiated, roadmap created  

---

## Overview

Initiated comprehensive research phase for improving the cherry pit classification model. User prioritized Option A (3-class training strategy) then Option C (SOTA architecture exploration). Created experimental roadmap and began deep research into modern architectures and three-class classification approaches.

---

## Key Constraints Identified

From project documentation review:

| Constraint | Value | Source |
|-----------|-------|--------|
| **Latency Budget** | <30ms for classification | architecture-quick-reference.md:139 |
| **Current Baseline** | ~16ms (ResNet50) | architecture-quick-reference.md:139 |
| **ResNet18 Speed** | ~8-10ms estimated | LESSONS.md:177 |
| **Training Infrastructure** | Google Colab Pro (Tesla T4) | LESSONS.md:114 |
| **Training Time** | ~30s/epoch, 30 epochs standard | MODEL_EXPERIMENTS.md:82 |
| **Critical Blocker** | No labeled "maybe" data | User confirmed |
| **Inference Hardware** | NVIDIA GPU (production), CPU (dev benchmarks) | architecture-quick-reference.md:139, HARDWARE_SPECIFICATIONS.md |

---

## Deliverables Created

### 1. EXPERIMENT_ROADMAP.md
**Location:** `docs/reference/EXPERIMENT_ROADMAP.md`

**Content:**
- 4-phase experimental plan (Research → Design → Implementation → Evaluation)
- Detailed breakdown of each phase with deliverables and timelines
- 7-week total timeline estimate
- Success criteria (target: >94.05% accuracy or 3-class model >92%)
- Risk assessment and mitigation strategies
- 5 preliminary experiment candidates with priorities

**Key Experiments Identified:**
1. Threshold Optimization (HIGH) - immediate value on existing 94.05% model
2. ConvNeXt-Tiny Training (HIGH) - modernized architecture
3. EfficientNet-B2 Retest (HIGH) - previous B0 failure may be normalization-related
4. ResNet50 + Attention (MEDIUM) - quick enhancement
5. Mixup 3-Class Training (MEDIUM) - synthetic "maybe" data

### 2. SOTA_RESEARCH.md (In Progress)
**Location:** `docs/reference/SOTA_RESEARCH.md`

**Content:**
- Architecture research on ConvNeXt, EfficientNet, MobileViT
- Three-class classification strategies comparison
- Training methodology enhancements (label smoothing, TTA)
- Preliminary architecture recommendations (Tier 1, 2, 3)
- Open questions and next research steps
- 10 relevant paper references

**Key Findings:**

**Top Architecture Candidates:**
| Model | Params | Why Test |
|-------|--------|----------|
| ConvNeXt-Tiny | 28M | Modernized ResNet, PyTorch native |
| EfficientNet-B2/B3 | 9-12M | Compound scaling, retest with unnormalized |
| ResNet50 + SE | ~26M | Minimal change, proven technique |

**3-Class Strategies Ranked:**
1. **Threshold Calibration** (Recommended) - post-hoc threshold optimization
2. **Synthetic Data (Mixup)** - generate "maybe" samples via interpolation
3. **Label Smoothing** - soft labels for uncertainty
4. **Evidential DL** - high complexity, last resort

---

## Research Conducted

### Architecture Papers Reviewed
1. **ConvNeXt (CVPR 2022)** - "A ConvNet for the 2020s"
   - Pure ConvNet competing with Transformers
   - Modernized design: depthwise conv, LayerNorm, GELU
   - ConvNeXt-Tiny: 28M params, promising candidate

2. **EfficientNet (ICML 2019)** - Compound scaling
   - Systematic depth/width/resolution scaling
   - B0-B7 family (5.3M to 66M params)
   - Previous B0 test: 92.66% (may be due to normalization mismatch)

3. **MobileViT (ICLR 2022)** - Hybrid CNN-Transformer
   - 5.6M params, designed for mobile/CPU
   - Local-global-local (LGL) processing
   - Novel but may struggle with small datasets

### Web Resources Accessed
- Papers with Code trending papers feed
- arXiv abstracts for architecture papers
- Hugging Face paper summaries

---

## Current Baseline Status

| Model | Accuracy | Latency | Params | Size | Class Support |
|-------|----------|---------|--------|------|---------------|
| **Best Training** | 94.05% | ~17ms | 25.6M | 90MB | 2-class |
| **Production** | 92.99% | ~16ms | 25.6M | 90MB | 3-class (2-stage) |
| **ResNet18 Alt** | 91.92% | ~8-10ms | 11.7M | 43MB | 2-class |

**Gap:** 2-class training vs 3-class production requirement

---

## Next Actions Identified

### Immediate (Next Session)
1. Complete attention mechanism research (CBAM vs SE vs ECA)
2. Verify PyTorch model availability for top candidates
3. Inventory exact training data counts (clean/pit split)
4. Document production threshold logic precisely

### Short-term (This Week)
5. Finish Phase 1 research document
6. Move to Phase 2 (Design & Prioritization)
7. Prepare Colab notebooks for top 3 experiments

### Pending Decisions
- Continue full research or start experiments immediately?
- Priority: accuracy vs speed vs 3-class fix?
- Compute budget: conservative (3-4 exps) or exploratory (8-10)?

---

## Open Questions from User

User indicated preference for:
- ✅ Option A first (3-class strategy), then Option C (architecture exploration)
- ✅ Include explicit brainstorming and research phase
- ❓ Priority between accuracy, speed, and 3-class fix (awaiting decision)
- ❓ Compute budget preference (awaiting decision)

---

## References

### Key Documents Referenced
- `docs/reference/MODEL_EXPERIMENTS.md` - Previous experiment results
- `docs/reference/architecture-quick-reference.md` - System timing constraints
- `docs/reference/LESSONS.md` - Historical insights and pitfalls
- `docs/reference/TRAINING_METHODOLOGY.md` - 2-stage approach critique

### Architecture Papers
1. Liu et al., "A ConvNet for the 2020s," CVPR 2022
2. Tan & Le, "EfficientNet: Rethinking Model Scaling," ICML 2019
3. Mehta & Rastegari, "MobileViT," ICLR 2022

---

**Session Status:** Phase 1 research 50% complete  
**Next Session:** Continue research or begin experiments (pending user decision)
