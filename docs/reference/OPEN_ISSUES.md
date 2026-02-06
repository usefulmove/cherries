---
name: Open Issues
status: Active
updated: 2025-02-05
---

# Open Issues & Technical Debt

This document tracks active issues, blockers, and technical debt per the enso protocol (Section 12 - Document Lifecycle).

**Last Updated:** 2025-02-05  
**Status:** Active items listed below, resolved items moved to LESSONS.md

---

## Issue 1: Git Large File Management

**Status:** Partially Resolved  
**Impact:** Blocks git push if not handled  
**Priority:** High

### Problem
GitHub has 100MB file size limit. Several files exceed this:
- Rosbag files (.db3): up to 742MB
- VimbaX SDK: 121MB
- Model files (.pt): ~1.1GB total (already in .gitignore)

### Current Solution
- Added `*.db3` pattern to .gitignore (excludes all rosbags)
- Added `threading_ws/src/cameras/include/VimbaX_2023-4/` to .gitignore
- Model files already excluded via existing `*.pt` rule

### Impact
- Test data (rosbags) not in version control
- VimbaX SDK must be installed separately from Allied Vision
- Model weights (~1.1GB) must be copied from backup drive: `/media/dedmonds/Extreme SSD/traina cherry line/threading_ws/src/cherry_detection/resource/`

### Future Resolution Options

| Option | Cost | Benefit | Recommendation |
|:-------|:-----|:--------|:---------------|
| Git LFS | ~$5/month for 50GB | Full version control, track history | Consider if models update frequently |
| External storage | Free | No repo bloat, simple | Document paths in README |
| Compressed rosbags | Medium effort | May fit under 100MB | Investigate for critical test data |
| Docker with pre-installed SDK | Low | Reproducible builds | Good for CI/CD |

### Next Actions
- [ ] Document VimbaX SDK installation steps in README
- [ ] Create script to sync model files from backup drive
- [ ] Decide on Git LFS adoption (cost vs benefit)

---

## Issue 2: Stem Detection Purpose Unknown

**Status:** Active Investigation  
**Impact:** Unknown - model loaded but practical usage unclear  
**Related:** [open-questions-stem-detection.md](./open-questions-stem-detection.md)  
**Priority:** Medium

### Key Questions

1. **Practical Purpose:** Does stem detection trigger sorting decisions? Are cherries with stems routed differently? Is this for quality grading or data collection?

2. **Robot Integration:** How does the robot/pneumatic system handle Type 6 (stem) messages? Is there a separate actuator?

3. **Performance Impact:** What's the latency cost vs 2-model pipeline? Could it run asynchronously?

4. **Training Data:** ~570 stem images exist (20240923 stems/), but training methodology unknown.

5. **3-class Coexistence:** Is the 3-class classifier (clean/maybe/pit) designed for stem-affected cherries?

6. **Fallback Behavior:** Error handling if stem model fails? Graceful degradation?

### Context
The production system (algorithm v6/hdr_v1) loads and executes the stem model (`stem_model_10_5_2024.pt`, 166MB) on every image. Detected stems are assigned Type 6 in Cherry messages and visualized with black bounding boxes. However, the practical impact on sorting decisions is unclear.

### Next Actions
- [ ] Interview system operators about stem detection in practice
- [ ] Review robot controller code for Type 6 handling
- [ ] Analyze production logs for stem detection frequency
- [ ] Benchmark latency: v6 (3-model) vs v7 (without stem)

---

## Issue 3: CIPster Nested Git Repository

**Status:** Resolved  
**Date:** 2025-02-05  
**Priority:** Completed

### Problem
Nested `.git` repository found at `threading_ws/src/plc_eip/include/CIPster/.git` (CIPster EtherNet/IP library from https://github.com/liftoff-sr/CIPster.git)

### Solution Applied
- Removed nested `.git` directory
- Added CIPster build artifacts to `.gitignore`
- Now tracked as regular source files (~2MB, safe to include)

### Result
CIPster library is now part of the parent repo without submodule complexity.

---

## Issue 4: Legacy System Maintenance

**Status:** Ongoing  
**Document:** [MIGRATION_cherry_system_to_threading_ws.md](./MIGRATION_cherry_system_to_threading_ws.md)  
**Priority:** Low

### Context
Two systems coexist in repository:
- `cherry_system/` - Legacy 2-model pipeline, simpler interfaces, Python orchestration
- `threading_ws/` - Production 3-model pipeline, HDR interfaces, C++ orchestration

### Maintenance Notes
- `cherry_system/` is read-only archive (reference only)
- `threading_ws/` receives updates and is the production system
- Differences documented in migration guide

### Next Actions
- [ ] Archive cherry_system documentation (mark as legacy)
- [ ] Ensure all new development happens in threading_ws

---

## Issue 5: Model File Synchronization

**Status:** Active  
**Impact:** Build/runtime failures if models missing  
**Priority:** Medium

### Problem
Model files (~1.1GB) are excluded from git but required for runtime. Located on external drive:
```
/media/dedmonds/Extreme SSD/traina cherry line/threading_ws/src/cherry_detection/resource/
```

### Current Workaround
Manual copy from backup drive when setting up new workspace.

### Production Models (Must Have)
| Model | Size | Purpose |
|:------|:-----|:--------|
| seg_model_red_v1.pt | 169M | Segmentation (Mask R-CNN) |
| classification-2_26_2025-iter5.pt | 91M | Classification 3-class |
| stem_model_10_5_2024.pt | 166M | Stem detection |

### Options
1. **Setup script:** `scripts/setup_models.sh` to copy from backup
2. **Git LFS:** Track models in git (costs money)
3. **S3/Artifactory:** Host models externally (requires auth)
4. **Keep external:** Document path, manual copy (current)

### Next Actions
- [ ] Create setup script for model synchronization
- [ ] Document model locations in README
- [ ] Consider Git LFS if models update frequently

---

## How to Update This Document

Per enso protocol (AGENTS.md):

1. **Add new issues** when discovered
2. **Mark resolved** with date and move learnings to LESSONS.md
3. **Update status** regularly during work
4. **Prune stale entries** - git preserves history

**Template for new issues:**
```markdown
## Issue N: [Title]

**Status:** [Active/Investigating/Resolved]  
**Impact:** [Description]  
**Priority:** [High/Medium/Low]

### Problem
[Description]

### Current Solution
[If any]

### Next Actions
- [ ] Action item 1
- [ ] Action item 2
```
