# Traina Project Migration Plan: enso v0.1.x → v0.2.0

## Executive Summary

This plan details the migration of the Traina project from enso protocol v0.1.x to v0.2.0. The primary changes involve expanding from 4 to 6 operations, restructuring the architecture documentation location, adding framework documentation organization, and creating new documentation files.

**Estimated Total Effort:** 4-6 hours
**Risk Level:** Low (documentation reorganization only)
**Rollback Strategy:** Git history preserved; can revert individual commits

---

## Current vs Target State

### Directory Structure Changes

```
BEFORE (v0.1.x):
├── AGENTS.md                          (4 operations)
├── docs/
│   ├── architecture/                  ← Moving to docs/core/
│   │   ├── INDEX.md
│   │   ├── hardware_io/ARCHITECTURE.md
│   │   ├── inference_pipeline/
│   │   │   ├── ARCHITECTURE.md
│   │   │   └── RESNET50_ANALYSIS.md
│   │   ├── system_overview/ARCHITECTURE.md
│   │   ├── tracking_orchestration/ARCHITECTURE.md
│   │   └── vision_acquisition/
│   │       ├── ARCHITECTURE.md
│   │       └── VIMBA_REFERENCE.md
│   ├── core/
│   │   ├── OPERATIONS.md
│   │   ├── PRD.md
│   │   └── STANDARDS.md
│   ├── logs/
│   ├── reference/
│   │   ├── completed/
│   │   ├── training/
│   │   │   └── pytorch/
│   │   │       ├── PYTORCH_01_OVERVIEW.md  ← Moving to docs/core/framework/pytorch/
│   │   │       ├── PYTORCH_02_MODELS.md
│   │   │       ├── PYTORCH_03_GPU_TENSORS.md
│   │   │       ├── PYTORCH_04_PREPROCESSING.md
│   │   │       ├── PYTORCH_05_INFERENCE.md
│   │   │       ├── PYTORCH_06_FUNCTIONAL.md
│   │   │       ├── PYTORCH_07_TRAINING.md
│   │   │       ├── PYTORCH_08_POSTPROCESSING.md
│   │   │       └── PYTORCH_09_COMPLETE_PIPELINE.md
│   │   └── [other docs]
│   ├── skills/
│   │   ├── benchmark-latency/
│   │   └── evaluate-model/
│   └── stories/

AFTER (v0.2.0):
├── AGENTS.md                          (6 operations, retrieval-led)
├── docs/
│   ├── core/
│   │   ├── PRD.md                     (unchanged)
│   │   ├── STANDARDS.md               (unchanged)
│   │   ├── OPERATIONS.md              (unchanged)
│   │   └── architecture/              ← NEW LOCATION (moved from docs/architecture/)
│   │       ├── INDEX.md
│   │       ├── hardware_io/
│   │       │   └── ARCHITECTURE.md
│   │       ├── inference_pipeline/
│   │       │   ├── ARCHITECTURE.md
│   │       │   └── RESNET50_ANALYSIS.md
│   │       ├── system_overview/
│   │       │   └── ARCHITECTURE.md
│   │       ├── tracking_orchestration/
│   │       │   └── ARCHITECTURE.md
│   │       └── vision_acquisition/
│   │           ├── ARCHITECTURE.md
│   │           └── VIMBA_REFERENCE.md
│   │
│   │   └── framework/                 ← NEW (moved from docs/reference/training/)
│   │       └── pytorch/
│   │           ├── INDEX.md           ← NEW (framework documentation index)
│   │           ├── PYTORCH_01_OVERVIEW.md
│   │           ├── PYTORCH_02_MODELS.md
│   │           ├── PYTORCH_03_GPU_TENSORS.md
│   │           ├── PYTORCH_04_PREPROCESSING.md
│   │           ├── PYTORCH_05_INFERENCE.md
│   │           ├── PYTORCH_06_FUNCTIONAL.md
│   │           ├── PYTORCH_07_TRAINING.md
│   │           ├── PYTORCH_08_POSTPROCESSING.md
│   │           └── PYTORCH_09_COMPLETE_PIPELINE.md
│   │
│   ├── reference/
│   │   ├── LESSONS.md                 ← NEW (lessons learned documentation)
│   │   ├── completed/
│   │   └── [other docs - training/pytorch removed]
│   │
│   ├── skills/
│   │   ├── benchmark-latency/
│   │   ├── evaluate-model/
│   │   └── training-colab/            ← NEW (training on Colab skill)
│   │       └── SKILL.md
│   │
│   ├── stories/
│   └── logs/
```

---

## Detailed Implementation Steps

### Phase 1: Foundation Updates (Step 1-2)
**Duration:** 1-1.5 hours  
**Dependencies:** None

---

#### Step 1: Update AGENTS.md to v0.2.0
**File:** `AGENTS.md`  
**Effort:** 45-60 minutes  
**Dependencies:** None  
**Type:** Major update

**Changes Required:**

1. **Update Principles Section** (Lines 17-32)
   - Change "Four operations" to "Six operations"
   - Add two new operations to the table:
     - **Retrieve** | Access external data sources, knowledge bases
     - **Transform** | Convert between formats, restructure content

2. **Update Terminology Section** (Lines 34-64)
   - Add definitions for Retrieve and Transform operations
   - Add definition for "Retrieval-led instruction" (new pattern in v0.2.0)

3. **Update Directory Structure** (Lines 66-81)
   - Update architecture path from `docs/architecture/` to `docs/core/architecture/`
   - Update skills path example to reflect training-colab skill
   - Add framework documentation location:
     ```
     docs/
       core/
         framework/     # Framework docs and tutorials
           pytorch/
             INDEX.md
     ```

4. **Update Bootstrapping Section** (Lines 83-109)
   - Step 4: Update path from `docs/architecture/INDEX.md` to `docs/core/architecture/INDEX.md`
   - Add reference to LESSONS.md creation in step 2
   - Add framework documentation bootstrap step

5. **Update Document Lifecycle Section** (Lines 110-140)
   - Add LESSONS.md to Reference section:
     "- LESSONS.md: Project learnings, retrospective insights"
   - Add framework documentation maintenance notes

6. **Update Discovery Protocol Section** (Lines 141-149)
   - Update reference from `docs/architecture/INDEX.md` to `docs/core/architecture/INDEX.md`

7. **Add New Section: Retrieval-Led Instruction** (After line 208)
   ```markdown
   ## 9. Retrieval-Led Instruction
   
   When working with complex information, use retrieval patterns:
   
   1. **Query** - Formulate specific information needs
   2. **Retrieve** - Access external sources, documentation, prior work
   3. **Synthesize** - Combine retrieved information with current context
   4. **Act** - Apply insights to current task
   
   Sources to retrieve from:
   - `docs/reference/LESSONS.md` - Prior project learnings
   - `docs/core/framework/` - Framework-specific documentation
   - Git history and previous implementations
   - External documentation (web, APIs)
   ```

8. **Renumber Sections**
   - Existing Section 9 (Skills) becomes Section 10
   - Existing Section 10 (Compaction) becomes Section 11
   - Existing Section 11 (Templates) becomes Section 12
   - Existing Section 12 (Agent Guidelines) becomes Section 13

9. **Update Context Scope Example** (Line 163)
   - Change: `docs/architecture/auth_layer/ARCHITECTURE.md`
   - To: `docs/core/architecture/auth_layer/ARCHITECTURE.md`

10. **Update Templates Section** (New Section 12)
    - Add LESSONS.md template
    - Update Architecture Module template to reflect new location
    - Add framework documentation template

**Verification:**
- All 6 operations documented
- All paths reference docs/core/architecture/
- No broken internal references

---

#### Step 2: Create LESSONS.md
**File:** `docs/reference/LESSONS.md`  
**Effort:** 30-45 minutes  
**Dependencies:** None  
**Type:** New file

**Content Structure:**
```markdown
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

### [YYYY-MM-DD] Category: Title

**Context:**
Brief description of the situation or problem encountered.

**What We Did:**
Description of the approach or solution implemented.

**Outcome:**
Results, metrics, or qualitative assessment of the outcome.

**Key Takeaway:**
The core insight or principle to remember.

**References:**
- Link to related story or documentation
- Link to relevant code or configuration

---

*Use the format above for new lessons. Keep entries concise and actionable.*
```

**Verification:**
- File created at correct path
- Frontmatter present
- Template structure clear and usable

---

### Phase 2: Architecture Migration (Steps 3-6)
**Duration:** 1.5-2 hours  
**Dependencies:** Phase 1 complete

---

#### Step 3: Create New Directory Structure
**Files:** Directory creation  
**Effort:** 10 minutes  
**Dependencies:** None  
**Type:** Infrastructure

**Commands:**
```bash
mkdir -p docs/core/architecture/{hardware_io,inference_pipeline,system_overview,tracking_orchestration,vision_acquisition}
mkdir -p docs/core/framework/pytorch
mkdir -p docs/skills/training-colab
```

**Verification:**
- All 5 architecture layer directories exist under docs/core/architecture/
- Framework directory exists
- Skills directory exists

---

#### Step 4: Move Architecture Files
**Files:** 8 files  
**Effort:** 15 minutes  
**Dependencies:** Step 3  
**Type:** File move

**Moves Required:**

| Source | Destination |
|--------|-------------|
| `docs/architecture/INDEX.md` | `docs/core/architecture/INDEX.md` |
| `docs/architecture/hardware_io/ARCHITECTURE.md` | `docs/core/architecture/hardware_io/ARCHITECTURE.md` |
| `docs/architecture/inference_pipeline/ARCHITECTURE.md` | `docs/core/architecture/inference_pipeline/ARCHITECTURE.md` |
| `docs/architecture/inference_pipeline/RESNET50_ANALYSIS.md` | `docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md` |
| `docs/architecture/system_overview/ARCHITECTURE.md` | `docs/core/architecture/system_overview/ARCHITECTURE.md` |
| `docs/architecture/tracking_orchestration/ARCHITECTURE.md` | `docs/core/architecture/tracking_orchestration/ARCHITECTURE.md` |
| `docs/architecture/vision_acquisition/ARCHITECTURE.md` | `docs/core/architecture/vision_acquisition/ARCHITECTURE.md` |
| `docs/architecture/vision_acquisition/VIMBA_REFERENCE.md` | `docs/core/architecture/vision_acquisition/VIMBA_REFERENCE.md` |

**Commands:**
```bash
cp docs/architecture/INDEX.md docs/core/architecture/INDEX.md
cp docs/architecture/hardware_io/ARCHITECTURE.md docs/core/architecture/hardware_io/ARCHITECTURE.md
cp docs/architecture/inference_pipeline/ARCHITECTURE.md docs/core/architecture/inference_pipeline/ARCHITECTURE.md
cp docs/architecture/inference_pipeline/RESNET50_ANALYSIS.md docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md
cp docs/architecture/system_overview/ARCHITECTURE.md docs/core/architecture/system_overview/ARCHITECTURE.md
cp docs/architecture/tracking_orchestration/ARCHITECTURE.md docs/core/architecture/tracking_orchestration/ARCHITECTURE.md
cp docs/architecture/vision_acquisition/ARCHITECTURE.md docs/core/architecture/vision_acquisition/ARCHITECTURE.md
cp docs/architecture/vision_acquisition/VIMBA_REFERENCE.md docs/core/architecture/vision_acquisition/VIMBA_REFERENCE.md
```

**Note:** Copy first, verify, then remove old files in Step 7.

**Verification:**
- All files copied to new locations
- File contents unchanged

---

#### Step 5: Update Architecture INDEX.md References
**File:** `docs/core/architecture/INDEX.md`  
**Effort:** 20 minutes  
**Dependencies:** Step 4  
**Type:** Reference update

**Changes Required:**

1. **Update Module Directory Table** (Lines 36-40)
   - Change paths in Directory column from `docs/architecture/...` to `docs/core/architecture/...`
   
   Before:
   ```markdown
   | Layer | Responsibility | Directory |
   |-------|----------------|-----------|
   | **[System Overview](./system_overview/ARCHITECTURE.md)** | ... | `docs/architecture/system_overview/` |
   ```
   
   After:
   ```markdown
   | Layer | Responsibility | Directory |
   |-------|----------------|-----------|
   | **[System Overview](./system_overview/ARCHITECTURE.md)** | ... | `docs/core/architecture/system_overview/` |
   ```

2. **Update Navigation Links** (Lines 14-30)
   - Mermaid diagram links remain relative, no change needed
   - But add note about new location

**Verification:**
- All directory references updated
- Relative links remain functional

---

#### Step 6: Update All Architecture Discovery Links
**Files:** 5 files  
**Effort:** 30 minutes  
**Dependencies:** Step 4  
**Type:** Reference update

**Files to Update:**

1. **docs/core/architecture/system_overview/ARCHITECTURE.md**
   - Update Discovery Links section (Line 37)
   - Change: `[Global Standards](../../core/STANDARDS.md)`
   - To: `[Global Standards](../../../core/STANDARDS.md)` (add extra ../)

2. **docs/core/architecture/hardware_io/ARCHITECTURE.md**
   - Update all relative links to other architecture layers
   - Change `../` references to reflect new depth

3. **docs/core/architecture/vision_acquisition/ARCHITECTURE.md**
   - Update discovery links
   - Update VIMBA_REFERENCE.md link

4. **docs/core/architecture/inference_pipeline/ARCHITECTURE.md**
   - Update discovery links
   - Update RESNET50_ANALYSIS.md link

5. **docs/core/architecture/tracking_orchestration/ARCHITECTURE.md**
   - Update discovery links

**Pattern for Link Updates:**
- Old: `../../core/STANDARDS.md` (from docs/architecture/layer/)
- New: `../../../core/STANDARDS.md` (from docs/core/architecture/layer/)

**Verification:**
- Check each file for broken relative links
- Verify discovery links work

---

### Phase 3: Framework Documentation Migration (Steps 7-9)
**Duration:** 1-1.5 hours  
**Dependencies:** Phase 1 complete

---

#### Step 7: Move PyTorch Documentation
**Files:** 9 files  
**Effort:** 15 minutes  
**Dependencies:** Step 3  
**Type:** File move

**Moves Required:**

| Source | Destination |
|--------|-------------|
| `docs/reference/training/pytorch/PYTORCH_01_OVERVIEW.md` | `docs/core/framework/pytorch/PYTORCH_01_OVERVIEW.md` |
| `docs/reference/training/pytorch/PYTORCH_02_MODELS.md` | `docs/core/framework/pytorch/PYTORCH_02_MODELS.md` |
| `docs/reference/training/pytorch/PYTORCH_03_GPU_TENSORS.md` | `docs/core/framework/pytorch/PYTORCH_03_GPU_TENSORS.md` |
| `docs/reference/training/pytorch/PYTORCH_04_PREPROCESSING.md` | `docs/core/framework/pytorch/PYTORCH_04_PREPROCESSING.md` |
| `docs/reference/training/pytorch/PYTORCH_05_INFERENCE.md` | `docs/core/framework/pytorch/PYTORCH_05_INFERENCE.md` |
| `docs/reference/training/pytorch/PYTORCH_06_FUNCTIONAL.md` | `docs/core/framework/pytorch/PYTORCH_06_FUNCTIONAL.md` |
| `docs/reference/training/pytorch/PYTORCH_07_TRAINING.md` | `docs/core/framework/pytorch/PYTORCH_07_TRAINING.md` |
| `docs/reference/training/pytorch/PYTORCH_08_POSTPROCESSING.md` | `docs/core/framework/pytorch/PYTORCH_08_POSTPROCESSING.md` |
| `docs/reference/training/pytorch/PYTORCH_09_COMPLETE_PIPELINE.md` | `docs/core/framework/pytorch/PYTORCH_09_COMPLETE_PIPELINE.md` |

**Commands:**
```bash
cp docs/reference/training/pytorch/*.md docs/core/framework/pytorch/
```

**Verification:**
- All 9 files copied
- Remove empty training/pytorch directory in cleanup step

---

#### Step 8: Create PyTorch Framework Index
**File:** `docs/core/framework/pytorch/INDEX.md`  
**Effort:** 30-45 minutes  
**Dependencies:** Step 7  
**Type:** New file

**Content Structure:**
```markdown
---
type: framework_index
framework: PyTorch
description: Comprehensive PyTorch documentation and tutorials for the Cherry Processing System.
---

# PyTorch Framework Documentation

## Overview

This directory contains structured PyTorch documentation covering the complete machine learning pipeline for cherry processing—from tensor basics to production deployment.

## Documentation Index

### Getting Started
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [01. Overview](PYTORCH_01_OVERVIEW.md) | System architecture, two-stage pipeline | None |
| [02. Models](PYTORCH_02_MODELS.md) | Model definitions, loading patterns | 01 |

### Core Concepts
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [03. GPU Tensors](PYTORCH_03_GPU_TENSORS.md) | Device management, tensor operations | 01-02 |
| [04. Preprocessing](PYTORCH_04_PREPROCESSING.md) | Image transforms, data loading | 01-03 |
| [05. Inference](PYTORCH_05_INFERENCE.md) | Model evaluation, prediction patterns | 01-04 |

### Advanced Topics
| Document | Topics | Prerequisites |
|----------|--------|---------------|
| [06. Functional API](PYTORCH_06_FUNCTIONAL.md) | Functional programming, nn.functional | 01-05 |
| [07. Training](PYTORCH_07_TRAINING.md) | Loss functions, optimizers, loops | 01-06 |
| [08. Postprocessing](PYTORCH_08_POSTPROCESSING.md) | Results handling, metrics | 01-07 |
| [09. Complete Pipeline](PYTORCH_09_COMPLETE_PIPELINE.md) | End-to-end integration | 01-08 |

## Quick Reference

### Common Patterns
```python
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model loading
model = torch.load('model.pt', map_location=device)
model.eval()

# Inference
def predict(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        return output.cpu()
```

### Key Code Locations
- `cherry_system/cherry_detection/cherry_detection/ai_detector.py` - Main inference implementation
- `training/scripts/` - Training and evaluation utilities

## Related Resources
- [System Architecture](../../architecture/inference_pipeline/ARCHITECTURE.md)
- [Training Colab Skill](../../../skills/training-colab/SKILL.md)
- [Model Evaluation Skill](../../../skills/evaluate-model/SKILL.md)
```

**Verification:**
- All 9 tutorial files linked correctly
- Navigation structure logical
- Related resources link to correct new paths

---

#### Step 9: Update PyTorch Doc Internal References
**Files:** 9 files  
**Effort:** 45-60 minutes  
**Dependencies:** Step 7  
**Type:** Reference update

**Files and Changes:**

1. **PYTORCH_01_OVERVIEW.md**
   - Update Next Section link (Line 131)
   - Change: `PYTORCH_02_MODELS.md`
   - To: `[02. Models](PYTORCH_02_MODELS.md)`

2. **PYTORCH_02_MODELS.md through PYTORCH_08_POSTPROCESSING.md**
   - Update all "Next Section" links at end of each file
   - Update any internal file references

3. **PYTORCH_09_COMPLETE_PIPELINE.md**
   - Update any concluding references
   - Add link back to INDEX.md

**Pattern:**
- All internal links should use relative markdown format: `[Section Title](FILENAME.md)`
- Links to framework index should work after Step 8

**Verification:**
- Test each Next Section link
- Verify no broken references

---

### Phase 4: Skills and Cleanup (Steps 10-13)
**Duration:** 1 hour  
**Dependencies:** Phase 1-3 complete

---

#### Step 10: Create training-colab Skill
**File:** `docs/skills/training-colab/SKILL.md`  
**Effort:** 30-45 minutes  
**Dependencies:** Step 3  
**Type:** New file

**Content Structure:**
```markdown
---
name: training-colab
description: Execute model training workflows on Google Colab with Google Drive integration.
---

# Training on Google Colab Skill

This skill enables efficient model training using Google Colab's GPU resources while maintaining code and data synchronization with the local project via Google Drive.

## When to Use

- Need GPU acceleration for model training
- Training experiments that exceed local hardware capabilities
- Long-running training jobs that benefit from cloud execution
- Testing hyperparameter variations at scale

## Prerequisites

1. Google account with Colab Pro (recommended) or Colab Free
2. Google Drive with project folder structure
3. Training data uploaded to Google Drive
4. Local project configured for Drive sync (see `docs/reference/colab-pro-setup.md`)

## Workflow

### Step 1: Prepare Training Configuration

Create or update a training config in `training/configs/`:

```yaml
# training/configs/experiment_name.yaml
model:
  name: resnet50
  num_classes: 2
  pretrained: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  
data:
  train_path: /content/drive/MyDrive/traina/data/train
  val_path: /content/drive/MyDrive/traina/data/val
  
augmentation:
  enabled: true
  normalize: false  # Set based on production requirements
```

### Step 2: Sync Code to Google Drive

```bash
# From project root
./training/scripts/sync_to_drive.sh
```

### Step 3: Launch Colab Notebook

1. Open `training/notebooks/colab_training.ipynb` in Google Colab
2. Connect to GPU runtime (Runtime → Change runtime type → GPU)
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Execute training cells

### Step 4: Download Results

After training completes:

```bash
# Sync trained models back from Drive
./training/scripts/sync_from_drive.sh \
  --source "drive/MyDrive/traina/experiments/" \
  --dest "training/experiments/"
```

## Configuration Options

### Environment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Development** | Quick iterations, small dataset | Testing configs, debugging |
| **Training** | Full dataset, GPU acceleration | Production model training |
| **Evaluation** | Validation/test metrics only | Assessing trained models |

### GPU Optimization

```python
# Enable mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Best Practices

1. **Version Control**
   - Git tracks code and configs locally
   - Drive stores datasets and model weights
   - Never commit large `.pt` files to git

2. **Experiment Tracking**
   - Use descriptive experiment names
   - Log all hyperparameters
   - Save training curves and metrics

3. **Data Management**
   - Keep training data in Drive under versioned folders
   - Use symbolic links for large datasets
   - Validate data integrity before training

4. **Cost Optimization**
   - Colab Free: Limited GPU hours per day
   - Colab Pro: Faster GPUs, longer sessions
   - Monitor runtime to avoid losing progress

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Drive not mounting | Re-authenticate, check permissions |
| CUDA out of memory | Reduce batch_size, use gradient accumulation |
| Training interrupted | Enable checkpointing, resume from last epoch |
| Slow data loading | Pre-cache dataset to local Colab storage |

## Related Resources

- [PyTorch Training Tutorial](../../core/framework/pytorch/PYTORCH_07_TRAINING.md)
- [Colab Pro Setup Guide](../../reference/colab-pro-setup.md)
- [Benchmark Latency Skill](../benchmark-latency/SKILL.md)
- [Evaluate Model Skill](../evaluate-model/SKILL.md)
```

**Verification:**
- Frontmatter present and valid
- All linked resources use correct paths
- Content is actionable and complete

---

#### Step 11: Cleanup Old Directories
**Files:** Directory removal  
**Effort:** 10 minutes  
**Dependencies:** Steps 4, 7 verified  
**Type:** Cleanup

**Commands:**
```bash
# Remove old architecture directory after verifying new location
rm -rf docs/architecture/

# Remove old training directory after verifying new location
rm -rf docs/reference/training/
```

**Verification:**
- All files exist in new locations
- No data loss
- Old directories removed

---

#### Step 12: Final Reference Updates
**Files:** Multiple reference docs  
**Effort:** 30 minutes  
**Dependencies:** All moves complete  
**Type:** Reference update

**Files to Check and Update:**

1. **docs/reference/system-architecture.md**
   - Check for references to docs/architecture/

2. **docs/reference/system-overview.md**
   - Check for architecture links

3. **docs/reference/colab-pro-setup.md**
   - Update any references to training/pytorch/

4. **docs/logs/*.md**
   - Check for references to moved files

5. **docs/reference/MODEL_EXPERIMENTS.md**
   - Update links to architecture and training docs

6. **docs/stories/STORY-003-Deployment-Readiness.md**
   - Update Context Scope references if needed

**Verification:**
- No references to old paths remain
- All documentation links functional

---

#### Step 13: Create Migration Summary Log
**File:** `docs/logs/2026-02-03-enso-migration-v0.2.0.md`  
**Effort:** 15 minutes  
**Dependencies:** All steps complete  
**Type:** Documentation

**Content Structure:**
```markdown
# Session: enso v0.1.x → v0.2.0 Migration

**Date:** 2026-02-03

## Overview

Completed migration of Traina project documentation structure from enso protocol v0.1.x to v0.2.0.

## Changes Made

### AGENTS.md Updates
- Expanded from 4 to 6 operations (added Retrieve, Transform)
- Added Retrieval-Led Instruction section
- Updated all directory references to docs/core/architecture/
- Renumbered sections to accommodate new content

### Architecture Restructure
- Moved docs/architecture/ → docs/core/architecture/
- Updated all discovery links in 5 architecture layer files
- Updated INDEX.md with new paths

### Framework Documentation
- Moved docs/reference/training/pytorch/ → docs/core/framework/pytorch/
- Created INDEX.md with navigation structure
- Updated internal references across all 9 tutorial files

### New Documentation
- Created docs/reference/LESSONS.md for project insights
- Created docs/skills/training-colab/SKILL.md for Colab workflows

## Files Modified
- AGENTS.md
- docs/core/architecture/INDEX.md
- docs/core/architecture/system_overview/ARCHITECTURE.md
- docs/core/architecture/hardware_io/ARCHITECTURE.md
- docs/core/architecture/vision_acquisition/ARCHITECTURE.md
- docs/core/architecture/inference_pipeline/ARCHITECTURE.md
- docs/core/architecture/tracking_orchestration/ARCHITECTURE.md
- docs/core/framework/pytorch/PYTORCH_01_OVERVIEW.md through PYTORCH_09_COMPLETE_PIPELINE.md
- docs/reference/colab-pro-setup.md (if applicable)
- docs/reference/system-architecture.md (if applicable)

## Files Created
- docs/reference/LESSONS.md
- docs/core/framework/pytorch/INDEX.md
- docs/skills/training-colab/SKILL.md

## Files Moved
- All 8 architecture files (copied then removed)
- All 9 PyTorch tutorial files (copied then removed)

## Files Removed
- docs/architecture/ (entire directory)
- docs/reference/training/ (entire directory)

## Verification Steps
- [ ] All 6 operations documented in AGENTS.md
- [ ] Architecture INDEX.md accessible at new path
- [ ] All 5 architecture layers have working discovery links
- [ ] PyTorch framework INDEX.md accessible
- [ ] All 9 PyTorch tutorials linked correctly
- [ ] LESSONS.md template ready for use
- [ ] training-colab skill frontmatter valid
- [ ] No broken references to old paths

## Next Steps
- Populate LESSONS.md with existing project insights
- Review and update any additional docs with architecture references
- Communicate new structure to team
```

**Verification:**
- Log file created
- All changes documented
- Checklist ready for final verification

---

## Complete File Path Update List

### Files Requiring Path Updates (Old → New)

| # | File | Old Path Reference | New Path Reference | Type |
|---|------|-------------------|-------------------|------|
| 1 | AGENTS.md | `docs/architecture/INDEX.md` | `docs/core/architecture/INDEX.md` | Document |
| 2 | AGENTS.md | `docs/architecture/layer/ARCHITECTURE.md` | `docs/core/architecture/layer/ARCHITECTURE.md` | Document |
| 3 | docs/architecture/INDEX.md → new | `docs/architecture/...` | `docs/core/architecture/...` | Internal links |
| 4 | docs/core/architecture/system_overview/ARCHITECTURE.md | `../../core/STANDARDS.md` | `../../../core/STANDARDS.md` | Relative link |
| 5 | docs/core/architecture/hardware_io/ARCHITECTURE.md | `../[other-layer]/` | `../../[other-layer]/` | Relative link |
| 6 | docs/core/architecture/vision_acquisition/ARCHITECTURE.md | `../` relative links | `../../` relative links | Relative link |
| 7 | docs/core/architecture/inference_pipeline/ARCHITECTURE.md | `../` relative links | `../../` relative links | Relative link |
| 8 | docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md | `../` relative links | `../../` relative links | Relative link |
| 9 | docs/core/architecture/tracking_orchestration/ARCHITECTURE.md | `../` relative links | `../../` relative links | Relative link |
| 10 | docs/core/architecture/vision_acquisition/VIMBA_REFERENCE.md | `../` relative links | `../../` relative links | Relative link |
| 11 | docs/core/framework/pytorch/INDEX.md | N/A | N/A | New file |
| 12 | docs/core/framework/pytorch/PYTORCH_01_OVERVIEW.md | N/A | N/A | Copied file |
| 13 | docs/core/framework/pytorch/PYTORCH_02-09.md | Internal tutorial links | Verify relative links | Tutorial series |
| 14 | docs/reference/colab-pro-setup.md | `training/pytorch/` | `core/framework/pytorch/` | Reference |
| 15 | docs/reference/system-architecture.md | `docs/architecture/` | `docs/core/architecture/` | Reference |
| 16 | docs/reference/system-overview.md | `docs/architecture/` | `docs/core/architecture/` | Reference |
| 17 | docs/skills/training-colab/SKILL.md | N/A | Links to PyTorch tutorials | New file |
| 18 | docs/stories/STORY-003-Deployment-Readiness.md | Check for architecture refs | Update if needed | Reference |
| 19 | docs/logs/*.md | Check for architecture refs | Update if needed | Reference |
| 20 | docs/reference/MODEL_EXPERIMENTS.md | Check for architecture refs | Update if needed | Reference |

---

## Dependency Graph

```
Phase 1: Foundation
├── Step 1: Update AGENTS.md (parallel with Step 2)
└── Step 2: Create LESSONS.md (parallel with Step 1)
    ↓
Phase 2: Architecture Migration
├── Step 3: Create directories
├── Step 4: Move architecture files (depends on 3)
├── Step 5: Update INDEX.md (depends on 4)
└── Step 6: Update layer discovery links (depends on 4)
    ↓
Phase 3: Framework Migration
├── Step 7: Move PyTorch docs (parallel with Phase 2)
├── Step 8: Create framework INDEX (depends on 7)
└── Step 9: Update PyTorch internal refs (depends on 7, 8)
    ↓
Phase 4: Skills and Cleanup
├── Step 10: Create training-colab skill (parallel)
├── Step 11: Cleanup old directories (depends on 4, 7 verification)
├── Step 12: Final reference updates (depends on all moves)
└── Step 13: Create migration log (depends on all)
```

---

## Risk Mitigation

### Data Loss Prevention
- **Strategy:** Copy files first, verify, then remove
- **Backup:** Git tracks all changes; can revert any step
- **Verification:** Checklist at end of each phase

### Reference Integrity
- **Strategy:** Grep for old paths before and after migration
- **Tool:** `grep -r "docs/architecture" --include="*.md"`
- **Tool:** `grep -r "docs/reference/training" --include="*.md"`

### Rollback Plan
1. Revert AGENTS.md changes: `git checkout AGENTS.md`
2. Restore architecture: `git checkout docs/architecture/`
3. Restore training docs: `git checkout docs/reference/training/`
4. Remove new files manually

---

## Success Criteria

- [ ] AGENTS.md contains 6 operations with definitions
- [ ] AGENTS.md has Retrieval-Led Instruction section
- [ ] All architecture files accessible at docs/core/architecture/
- [ ] All 5 architecture layers have working relative links
- [ ] PyTorch tutorials accessible at docs/core/framework/pytorch/
- [ ] PyTorch INDEX.md has navigation to all 9 tutorials
- [ ] LESSONS.md exists with template structure
- [ ] training-colab skill created with valid frontmatter
- [ ] No references to old paths remain in any .md files
- [ ] All internal links verified functional
- [ ] Migration log created documenting all changes

---

## Execution Checklist

### Phase 1: Foundation [ ]
- [ ] Step 1: AGENTS.md updated to v0.2.0
- [ ] Step 2: LESSONS.md created

### Phase 2: Architecture [ ]
- [ ] Step 3: New directory structure created
- [ ] Step 4: Architecture files moved
- [ ] Step 5: INDEX.md references updated
- [ ] Step 6: All layer discovery links updated

### Phase 3: Framework [ ]
- [ ] Step 7: PyTorch docs moved
- [ ] Step 8: PyTorch INDEX.md created
- [ ] Step 9: PyTorch internal references updated

### Phase 4: Skills & Cleanup [ ]
- [ ] Step 10: training-colab skill created
- [ ] Step 11: Old directories removed
- [ ] Step 12: All remaining references updated
- [ ] Step 13: Migration log created

### Final Verification [ ]
- [ ] All success criteria met
- [ ] Grep shows no old path references
- [ ] Spot-check random links
- [ ] Commit migration changes

---

*End of Migration Plan*
