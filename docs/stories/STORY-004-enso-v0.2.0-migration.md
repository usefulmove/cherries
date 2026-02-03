# STORY-004: Migrate to enso v0.2.0 Protocol

## Goal
Migrate the Traina project documentation structure from enso protocol v0.1.x to v0.2.0, adopting the 6 operations framework, consolidating architecture documentation to docs/core/, creating framework documentation index, and adding LESSONS.md for project insights.

## Acceptance Criteria
- [x] AGENTS.md updated to v0.2.0 (6 operations: Write, Select, Compress, Isolate, Probe, Assign + retrieval-led reasoning instruction)
- [x] docs/reference/LESSONS.md created with template structure for capturing project learnings
- [x] Architecture documentation moved from docs/architecture/ to docs/core/architecture/
- [x] All 5 architecture layer files updated with correct relative discovery links
- [x] PyTorch documentation moved to docs/core/framework/pytorch/
- [x] PyTorch INDEX.md created with navigation structure
- [x] docs/skills/training-colab/SKILL.md created for Colab training workflows
- [x] All internal references updated to reflect new paths
- [x] Old directories (docs/architecture/, docs/reference/training/) removed
- [x] Migration log created documenting all changes

## Context Scope
**Write:**
- AGENTS.md
- docs/reference/LESSONS.md
- docs/core/architecture/INDEX.md
- docs/core/architecture/*/ARCHITECTURE.md (5 files)
- docs/core/architecture/inference_pipeline/RESNET50_ANALYSIS.md
- docs/core/architecture/vision_acquisition/VIMBA_REFERENCE.md
- docs/core/framework/pytorch/INDEX.md
- docs/core/framework/pytorch/PYTORCH_*.md (9 files)
- docs/skills/training-colab/SKILL.md
- docs/logs/2026-02-03-enso-migration-v0.2.0.md

**Read:**
- docs/reference/ENSO_V0.2.0_MIGRATION_PLAN.md
- docs/core/PRD.md
- docs/core/STANDARDS.md
- docs/core/OPERATIONS.md
- Existing architecture files (before move)
- Existing PyTorch tutorial files (before move)

**Exclude:**
- Source code files (*.py)
- Training notebooks and scripts
- Model weights and data files

## Approach
Following the detailed migration plan at docs/reference/ENSO_V0.2.0_MIGRATION_PLAN.md:

1. **Phase 1: Foundation** - Update AGENTS.md with 6 operations and retrieval-led instruction; create LESSONS.md template
2. **Phase 2: Architecture Migration** - Create new structure, move files, update all discovery links
3. **Phase 3: Framework Docs** - Move PyTorch tutorials, create INDEX.md with navigation
4. **Phase 4: Skills & Cleanup** - Create training-colab skill, remove old directories, update remaining references, create migration log

## Notes
**Lessons to Capture:**
- Model denormalization issue affecting inference latency (from STORY-003)
- Unnormalized training approach for production compatibility
- Hybrid local/Colab workflow for GPU training
- Importance of retrieval-led reasoning for framework-specific tasks

**Framework Documentation Index Value:**
The 9 PyTorch tutorials form a complete curriculum. Having them in docs/core/framework/ with an INDEX.md makes them always-available and follows enso v0.2.0 best practices (100% retrieval accuracy vs 79% with on-demand skills).

**training-colab Skill Scope:**
This should capture the complete human-in-the-loop Colab Pro workflow: config preparation → Drive sync → notebook execution → results download. It's a recurring process, so Skill format (on-demand) is appropriate.
