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

*Use the format above for new lessons. Keep entries concise and actionable. Focus on transferable insights rather than situational details.*
