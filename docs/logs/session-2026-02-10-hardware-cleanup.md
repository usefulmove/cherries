# Session: Hardware Documentation & Model Cleanup
**Date:** 2026-02-10

## Overview
Addressed critical discrepancies in system documentation regarding hardware capabilities. Discovered that the production system utilizes an NVIDIA GPU, contrary to documentation stating "CPU only". Cleaned up temporary experimental files from Phase 2 and reorganized model artifacts.

## Key Accomplishments

### 1. Hardware Clarification
- **Discovery:** Production code (`ai_detector*.py`) explicitly checks for and uses CUDA, contradicting docs that claimed "CPU inference only".
- **Documentation Fix:** Updated 9 core documentation files (PRD, Architecture, Roadmap) to reflect:
    - **Production:** NVIDIA GPU enabled (Model TBD).
    - **Training:** Google Colab Pro.
    - **Baselines:** 16ms (CPU measurement) is a relative baseline, not a production limit.
- **Created:** `docs/reference/HARDWARE_SPECIFICATIONS.md` to serve as the single source of truth for compute infrastructure.

### 2. Requirement Refinement
- **Issue:** The "<30ms" latency target was found to be an arbitrary safety margin (approx 2x the 16ms CPU baseline) rather than a constraint derived from belt speed/physics.
- **Action:** Removed "30ms" hard target from documentation. Replaced with "Maintain ~16ms throughput (baseline)" to emphasize relative performance until production GPU benchmarks are available.

### 3. Model Organization & Cleanup
- **Moved:** ConvNeXt V2 artifacts from `temp-phase2-experiments/` to permanent home: `threading_ws/src/cherry_detection/resource/experimental/convnextv2/`.
- **Created:** `threading_ws/src/cherry_detection/resource/experimental/README.md` cataloging experimental models.
- **Archived:** Training notebook to `training/notebooks/archive/`.
- **Deleted:** `temp-phase2-experiments/` folder, recovering **4.7GB** of disk space.
- **Preserved:** `temp-old-training-docs/` as requested.

## Artifacts Modified
- `docs/core/PRD.md`
- `docs/core/architecture/system_overview/ARCHITECTURE.md`
- `docs/reference/architecture-quick-reference.md`
- `docs/reference/EXPERIMENT_ROADMAP.md`
- `docs/reference/LESSONS.md`
- `docs/reference/MODEL_EXPERIMENTS.md`
- `docs/reference/PHASE2_IMPLEMENTATION_SUMMARY.md`
- `docs/stories/STORY-006-Post-Phase2-Decision.md`
- `AGENTS.md`

## Artifacts Created
- `docs/reference/HARDWARE_SPECIFICATIONS.md`
- `threading_ws/src/cherry_detection/resource/experimental/README.md`
- `docs/logs/session-2026-02-10-hardware-cleanup.md`

## Next Steps
- Continued model training experimentation.
