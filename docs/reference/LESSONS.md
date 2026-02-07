---
type: lessons_learned
description: Project retrospective insights, patterns discovered, and accumulated wisdom.
---

# Project Lessons Learned

## Overview

This document captures insights, patterns, and retrospective learnings from the Traina project. It serves as institutional memory for future development decisions.

## Categories

### Performance
Insights about system performance, bottlenecks, and optimization strategies.

### Architecture
Patterns and anti-patterns discovered during system design and evolution.

### ML/Training
Lessons from model training, inference optimization, and data pipeline design.

### Development Workflow
Productivity patterns, tooling decisions, and process improvements.

---

## Lessons

### [2026-01-30] ML/Training: Model Weight Denormalization Causes Severe Latency Regression

**Context:**
After training a new ResNet50 model for cherry pit detection, initial CPU inference latency was ~168ms compared to the production baseline of ~16ms. This represented a 10x performance regression that would prevent deployment.

**What We Did:**
Discovered that model weights contained "denormal" values (subnormal floating-point numbers < 1e-35) causing CPU execution to fall back to slow software emulation paths. Applied a post-training fix to zero out denormal values in the model weights.

**Outcome:**
Inference latency dropped from ~168ms to ~16.7ms, matching the production baseline. Model accuracy remained at 94.05% (exceeding the 93% target).

**Key Takeaway:**
Always validate model weights for denormal values after training, especially when using mixed precision or training on GPUs. Denormals can silently destroy CPU inference performance.

**References:**
- [STORY-003: Deployment Readiness](../stories/STORY-003-Deployment-Readiness.md)
- [Model Experiments Log](./MODEL_EXPERIMENTS.md)

---

### [2026-01-30] ML/Training: Unnormalized Training Avoids ROS2 Node Modifications

**Context:**
The production ROS2 system feeds raw camera images (0-255 pixel values) directly to the model. Our best trained model expected ImageNet-normalized inputs (mean-subtracted, scaled), requiring code changes to the production inference pipeline.

**What We Did:**
Instead of modifying production code, we trained a new model variant with augmentation enabled but normalization disabled. The model learns to handle raw pixel distributions directly.

**Outcome:**
Achieved 94.05% accuracy with unnormalized inputs, enabling true "drop-in replacement" deployment without touching the ROS2 control node. Avoided production code changes and regression testing overhead.

**Key Takeaway:**
When deploying ML models to production systems with fixed preprocessing pipelines, consider training models that work with the existing data format rather than modifying the pipeline. The training cost is often lower than the deployment risk.

**References:**
- [STORY-003: Deployment Readiness](../stories/STORY-003-Deployment-Readiness.md)
- [Inference Pipeline Architecture](../core/architecture/inference_pipeline/ARCHITECTURE.md)

---

### [2026-02-03] Development Workflow: enso Protocol Migration Requires Careful Path Coordination

**Context:**
Migrating from enso v0.1.x to v0.2.0 involved moving architecture docs from `docs/architecture/` to `docs/core/architecture/` and creating framework documentation indices. This required updating dozens of internal cross-references.

**What We Did:**
Created a comprehensive migration plan (ENSO_V0.2.0_MIGRATION_PLAN.md) documenting all file moves and reference updates. Executed in 4 phases: Foundation → Architecture Migration → Framework Docs → Skills & Cleanup.

**Outcome:**
Successfully migrated 17 files (8 architecture + 9 PyTorch tutorials) while maintaining link integrity. Created LESSONS.md to capture this and future insights.

**Key Takeaway:**
Documentation restructuring requires systematic planning to avoid broken links and lost context. Treat documentation refactoring with the same rigor as code refactoring—use stories, track dependencies, and verify integrity at each step.

**References:**
- [Migration Plan](./ENSO_V0.2.0_MIGRATION_PLAN.md)
- [STORY-004: enso v0.2.0 Migration](../stories/STORY-004-enso-v0.2.0-migration.md)
- [Updated AGENTS.md](../../AGENTS.md)

---

### [2026-02-03] ML/Workflow: Notebook Drift requires Local Smoke Tests

**Context:**
A custom training loop implemented directly in a Colab notebook for differential learning rates diverged from the tested `src/` library code. Specifically, the notebook used a generic `f1_score` key while the library returned `f1`, causing a `KeyError` after the first epoch of training.

**What We Did:**
Fixed the key error and established a "Smoke Test" protocol: all notebook logic must be verifiable locally (on CPU) with a "Dry Run" mode (1 epoch, limited batches) before being handed off to Colab.

**Outcome:**
Prevented future "deploy-and-crash" cycles where simple logic errors waste developer time and GPU resources.

**Key Takeaway:**
Never re-implement core logic in notebooks if possible. If custom logic is needed, implement it in a script, verify it locally, and import it. Always "smoke test" the notebook locally before sync.

**References:**
- [Colab Optimization Experiments](../../training/notebooks/colab_optimization_experiments.ipynb)
- [Training on Colab Skill](../skills/training-colab/SKILL.md)

---

### [2026-02-04] ML/Training: Always Verify GPU Availability Before Training

**Context:**
Ran optimization experiments on Google Colab but the runtime was configured for CPU instead of GPU. Training took ~12 minutes per epoch (vs ~30 seconds with GPU), and the ResNet18 experiment never completed due to time constraints. The differential LR experiment completed but with potentially non-representative results.

**What We Did:**
1. Identified the issue from notebook output showing `CUDA available: False` and ~710s epoch times
2. Added a hard-stop GPU verification check to the training notebook
3. Documented the experiment as "inconclusive" in MODEL_EXPERIMENTS.md

**Outcome:**
~6 hours of CPU compute time produced inconclusive results. The ResNet18 experiment needs to be re-run with GPU enabled.

**Key Takeaway:**
Always verify GPU availability at the start of training notebooks. Add a hard-stop check that raises an error if CUDA is unavailable when GPU training is expected. The few seconds spent checking saves hours of wasted compute.

**Implementation:**
```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU required for training. "
        "Go to Runtime -> Change runtime type -> GPU"
    )
```

**References:**
- [Model Experiments Log](./MODEL_EXPERIMENTS.md)
- [Colab Optimization Experiments](../../training/notebooks/colab_optimization_experiments.ipynb)

---

### [2026-02-04] ML/Workflow: Notebook Configuration with Skip Flags for "Run All"

**Context:**
Optimization experiments in Colab often skip some experiments while running others (e.g., skip inconclusive Exp 1, run Exp 3). The notebook was failing with TypeError when trying to format skipped experiment results.

**What We Did:**
Implemented a configuration cell pattern with explicit skip flags:
```python
DRY_RUN = False                    # Smoke test mode
SKIP_EXPERIMENT_1 = True           # Skip inconclusive experiments
THRESHOLD_MODEL = 'production'     # Model selection for Exp 2
```
Each experiment cell checks its skip flag before executing. Fixed final summary to use `is not None` instead of checking `in locals()`.

**Outcome:**
Enables reliable "Run All" behavior where some experiments are cleanly skipped and won't cause downstream errors. Improves robustness of complex notebooks.

**Key Takeaway:**
When a notebook has conditional experiments, use explicit config flags and `is not None` checks rather than relying on variable presence detection. This survives refactoring and skipped cells better.

**References:**
- [Colab Optimization Experiments](../../training/notebooks/colab_optimization_experiments.ipynb)
- [Training on Colab Skill](../skills/training-colab/SKILL.md)

---

### [2026-02-04] ML/Training: ResNet18 Achieves 1% Accuracy Drop for 2x Model Compression

**Context:**
Hypothesis: ResNet18 (11.7M params) can match ResNet50 (25.6M params) accuracy with faster inference.

**What We Did:**
Trained ResNet18 with identical augmentation + unnormalized setup as ResNet50. Best epoch: 6 (91.92% accuracy).

**Outcome:**
- **ResNet18:** 91.92% accuracy, 43MB model, estimated 8-10ms latency (40-50% faster)
- **ResNet50 (our best):** 94.05% accuracy, 90MB model, ~17ms latency
- **Production (ResNet50):** 92.99% accuracy, 90MB model, ~16ms latency

**Key Takeaway:**
ResNet18 is acceptable for latency-constrained environments if 1% accuracy drop is tolerable. However, pit recall (89.58% vs ~93%) is lower—important for food safety. Decision hinges on priorities (speed vs. accuracy).

**References:**
- Epoch 6 metrics: 92.00% precision, 91.76% recall, 91.86% F1
- [Model Experiments Log](./MODEL_EXPERIMENTS.md) (Experiment Set 3)
- `training/experiments/resnet18_augmented_unnormalized/metrics.json`

---

### [2026-02-05] Documentation/Workflow: Consolidate Questions Before Critical Meetings

**Context:**
Preparing for a handoff meeting with the original engineer and Russ. Had accumulated 76 questions across multiple documents (classification-questions.md, developer-questions.md, possible-improvements.md), but this was too many for a productive meeting and had significant overlap.

**What We Did:**
1. Audited all existing question documents for duplicates and gaps
2. Consolidated into a single focused discussion guide (open-questions-20260205.md)
3. Narrowed from 76 questions to ~20 prioritized questions across 5 key topics
4. Verified "maybe" category handling through code review before the meeting
5. Archived redundant documents to docs/reference/completed/

**Outcome:**
Created a focused 130-line discussion guide with verified findings, ready for the 2pm handoff meeting. Eliminated duplicate questions and provided concrete context for each discussion topic.

**Key Takeaway:**
Before critical meetings, consolidate scattered questions into a single prioritized document. Remove duplicates, verify assumptions through code/docs review when possible, and focus on actionable questions that drive decisions. Archive rather than delete old versions to preserve history.

**Process:**
1. Gather all existing question lists
2. Identify duplicates and consolidate
3. Prioritize into Must Ask / Should Ask / Can Figure Out Later
4. Verify key technical assumptions before asking
5. Create focused discussion guide with context
6. Archive redundant documents

**References:**
- [Handoff Meeting Prep](../logs/2026-02-05-handoff-meeting-prep.md)
- [Discussion Guide](./open-questions-20260205.md)
- [Completed Questions](../reference/completed/)

---

### [2026-02-05] Architecture/Documentation: Document Operational Details in Architecture Guides

**Context:**
Discovered through code review that the "maybe" category (label 5) isn't just a classification output—it's a critical safety feature that triggers manual worker review via yellow projection highlights. This operational workflow wasn't captured in the architecture documentation, only the classification logic was mentioned.

**What We Did:**
1. Updated [Inference Pipeline Architecture](../core/architecture/inference_pipeline/ARCHITECTURE.md) with comprehensive classification categories table including:
   - All four labels (1=clean, 2=pit, 3=side, 5=maybe)
   - Threshold logic (≥0.75 pit, ≥0.5 maybe/clean)
   - Color coding for visualization
   - Safety feature explanation (manual review pathway)
   
2. Updated [Tracking & Orchestration Architecture](../core/architecture/tracking_orchestration/ARCHITECTURE.md) with:
   - Color coding table for projection system
   - Specific code references (helper.cpp lines 66-69, 115-134)
   - Safety workflow description
   - Cross-reference to inference pipeline docs

3. Added bidirectional discovery links between architecture documents

**Outcome:**
Architecture documentation now captures both the technical implementation AND the operational workflow. Future developers can understand not just "what" the categories are, but "why" they matter for system safety and "how" they're visualized for worker review.

**Key Takeaway:**
Architecture documentation should capture operational workflows, not just technical implementation. When a system feature has human-facing consequences (like visual highlights for manual review), document the full workflow across all relevant architecture layers. Use bidirectional discovery links to help agents navigate between related components.

**Implementation Pattern:**
- Technical details → Inference Pipeline layer
- Visualization/UX details → Tracking & Orchestration layer  
- Cross-reference both with discovery links
- Include specific code locations for agentic discovery

**References:**
- [Inference Pipeline](../core/architecture/inference_pipeline/ARCHITECTURE.md)
- [Tracking & Orchestration](../core/architecture/tracking_orchestration/ARCHITECTURE.md)
- Code: `ai_detector.py:246, 376-383` (thresholds), `helper.cpp:66-69, 115-134` (visualization)

---

### [2026-02-06] ML/Training: Two-Stage Training Methodology Has Documented Concerns

**Context:**
Discovered the production classification model (`classification-2_26_2025-iter5.pt`) was trained using a two-stage approach: Stage 1 trains binary classifier (pit vs no_pit), Stage 2 fine-tunes on misclassifications labeled "maybe" to create 3-class output. This creates the "maybe" class from model errors rather than ground truth labels.

**Assessment:**
This approach has critical flaws:
1. **Safety Risk:** Stage 1 misses become permanent false negatives
2. **Architecture Violation:** "Maybe" class violates mutual exclusivity; defined by errors, not ground truth  
3. **Catastrophic Forgetting:** High risk of eroding Stage 1 learning during Stage 2 fine-tuning
4. **Threshold Redundancy:** 3-class output converted back to thresholds (0.5/0.75) in production anyway
5. **Auditability:** Complex cascade difficult to validate and explain to regulators

**Recommended Alternatives:**
1. **Current 3-class explicit** (keep): Manually label "maybe" examples, train 3-class from scratch
2. **Enhanced 2-class**: Binary classifier with calibrated confidence tiers (pit≥0.85 auto-reject, 0.65-0.85 review, <0.65 accept)
3. **Ensemble methods**: Multiple models; disagreement indicates uncertainty

**Key Takeaway:**
Learning from model errors (two-stage) is inferior to explicit labeling or confidence-based routing. The "maybe" class should represent genuine uncertainty (lighting, occlusion, damage), not training artifacts. Training data should reflect ground truth, not algorithmic confusion.

**References:**
- [Training Methodology](./TRAINING_METHODOLOGY.md)
- [Inference Pipeline Architecture](../core/architecture/inference_pipeline/ARCHITECTURE.md)
- [Model Experiments](./MODEL_EXPERIMENTS.md)

---

*Use the format above for new lessons. Keep entries concise and actionable. Focus on transferable insights rather than situational details.*
