# Context Engineering Protocol

[enso – a context management structure for agentic coding](https://github.com/usefulmove/enso)

enso v0.2.0

A single-file seed for managing context across LLM agents, sessions, and tools.

---

**IMPORTANT:** Prefer retrieval-led reasoning over pre-training-led reasoning for framework-specific and domain-specific tasks.

## 1. Purpose

This protocol defines how to manage context when working with LLM agents on software projects. The workflow:

1. Drop this file into a project directory
2. Point an agent to it
3. Agent bootstraps the directory structure
4. Agent creates a PRD from conversation with human
5. Documents evolve as work unfolds, staying compact and focused

The goal: maintain the smallest set of high-signal tokens needed for the next step.

## 2. The Six Operations

**Context is finite.** The context window is the agent's working memory. Every token competes for attention. Treat context as a scarce resource.

| Operation | What It Does | Why It Matters |
|-----------|--------------|----------------|
| **Write** | Persist information outside the context window | Working memory is temporary; persistence survives sessions |
| **Select** | Load only what's needed right now | Don't waste tokens on irrelevant context |
| **Probe** | Actively search (grep, LSP, glob) for answers | Don't assume you know what's in the codebase |
| **Compress** | Summarize to fit the token budget | When context gets full, condense instead of dropping |
| **Isolate** | Split work across multiple scopes | Divide complex tasks to stay within limits |
| **Assign** | Choose the ideal agent for each task | Match task requirements to agent capabilities |

**Keep current, not historical.** Documents reflect the present state. Git preserves history. Don't accumulate cruft in docs.

**Progressive disclosure.** Load context only when needed. Frontmatter before full docs. Summaries before details.

## 3. Terminology

| Term | Definition |
|------|------------|
| **Working Context** | Active tokens in the current LLM call. Ephemeral, limited by context budget. |
| **Persistent Context** | Markdown documents that survive across sessions. Includes Core, Stories, Reference, Skills, and Logs. |
| **Reference Context** | Queryable external sources: codebase (LSP, grep), RAG indexes, web. Not stored in the doc system. |
| **Compaction** | Summarizing working context into persistent context before it's lost. |
| **Context Budget** | The token limit of the model's context window. |
| **Context Scope** | Per-story declaration of file boundaries: what the agent can write, read, or must exclude. |

**Hierarchy:**

```
WORKING CONTEXT (ephemeral, token-limited)
    |
    | <- Select (load)     -> Write (persist)
    v                         v
PERSISTENT CONTEXT (markdown, durable)
  |-- Core Docs (PRD, Architecture, Standards)
  |-- Stories (active tasks)
  |-- Reference (conventions, completed work)
  |-- Skills (on-demand capabilities)
  |-- Logs (session summaries)

REFERENCE CONTEXT (external, queryable)
  |-- Codebase (LSP, grep, git)
  |-- RAG indexes
  |-- External sources (web, APIs)
```

## 4. Directory Structure

```
docs/
  core/           # Source of Truth (PRD, Architecture)
    PRD.md        # Problem, goals, scope, requirements
    STANDARDS.md  # Coding conventions, patterns
    architecture/ # System design modules
      INDEX.md    # Entry point / Routing table
      .../        # Subdirectories per layer/module
    framework/    # Framework documentation
      pytorch/    # PyTorch tutorials and reference
        INDEX.md
  stories/        # Active Units of Work (The "Ticket" system)
  reference/      # Long-term Memory (Lessons, Conventions)
    completed/    # Archived stories
    LESSONS.md    # Project learnings and insights
  skills/         # Local Capabilities (Scripts, Tests)
  logs/           # Session History
```

## 5. Bootstrapping

When an agent encounters this file in a new project:

**Step 1: Create structure**
```bash
mkdir -p docs/{core,architecture,stories,reference/completed,skills,logs}
touch docs/reference/LESSONS.md
```

**Step 2: Add retrieval-led reasoning instruction** to root `AGENTS.md`:
```
IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning 
for framework-specific and domain-specific tasks.
```

**Step 3: Gather context** — Prompt the human for the problem, success criteria, scope, and constraints.

**Step 4: Generate PRD** — Create `docs/core/PRD.md` from the conversation.

**Step 5: System Mapping** — Probe the codebase to create `ARCHITECTURE.md`, identify capabilities, and document conventions.

**Step 6: First story** — Create the initial story in `docs/stories/`.

**Step 7: Begin work**

## 6. Document Lifecycle

Context is living code. Refactor documentation as aggressively as you refactor code. Stale context is technical debt.

**Core Docs** (PRD, Architecture, Standards) — Update in place when scope changes. Don't preserve history—git does.

**Stories** — Create when planning, update during execution, move to `reference/completed/` when done.

**Reference** — Conventions and lessons. Read-only during execution; prune when irrelevant. **Update `LESSONS.md` with new learnings.**

**Skills** — Add as needed, update when procedures change, remove when obsolete.

**Logs** — Append session summaries after compaction; prune when no longer informative.

## 7. Discovery Protocol

Agents must use the **Discoverable Architecture** pattern to avoid context overload.

1.  **Start at `docs/core/architecture/INDEX.md`**: Read the routing table to find the relevant system layer.
2.  **Drill Down**: Read the `ARCHITECTURE.md` in the specific layer's directory.
3.  **Follow Links**: Use the "Discovery Links" in that module to find specific code or reference docs.

**Do NOT** read the entire `docs/` folder into context. Be surgical.

## 8. Context Scope

Every story declares its context boundaries:

```markdown
## Context Scope

**Write** (files this task will modify):
- src/auth/login.ts
- src/auth/session.ts

**Read** (files for reference only):
- docs/core/ARCHITECTURE.md
- src/auth/types.ts

**Exclude** (ignore these):
- src/legacy/
- *.test.ts
```

**Enforcement:**
- Agent must not modify files outside Write scope
- Read files are a seed; use Probe to dynamically discover related context
- Agent should consult Read files before making changes
- Agent should avoid loading Excluded paths into context
- Scope changes require explicit human approval

## 9. Skills

Skills are discoverable, on-demand capabilities. They follow the [Agent Skills specification](https://agentskills.io/specification).

**Location:** `docs/skills/`

**Structure:**
```
docs/skills/
  skill-name/
    SKILL.md      # Required: frontmatter + when to use
    scripts/      # Optional: executable code
    references/   # Optional: additional docs
    assets/       # Optional: templates, data files
```

**Discovery:** Agent scans directories at session start, reads frontmatter (~100 tokens) for discovery, loads full skill only when needed.

## 9.1. Framework Documentation Index

Store version-matched framework docs in `docs/core/framework/` and add an index to `AGENTS.md`:

```markdown
## Framework Documentation
Location: docs/core/framework/

| Section | Files |
|---------|-------|
| Routing | routing.md, navigation.md |
| Caching | cache-directives.md, cache-lifecycle.md |
```

**Why this works:** Always present, standard Markdown, retrieval-led (100% accuracy vs. 79% with on-demand skills).

## 10. Compaction

Moves insights from working context to persistent context.

**Triggers:** ~80% token utilization, completing a story, circular conversation, ending session.

**Process:** Summarize decisions, list artifacts, extract lessons to `LESSONS.md`, write summary to `logs/`, continue with fresh context.

## 11. Templates

Templates are guidelines, not rigid forms. Start minimal, expand as needed.

### PRD

```markdown
# [Project] PRD

## Problem
What problem are we solving? Why does it matter?

## Goals
What does success look like?

## Scope
**In scope:** ...
**Out of scope:** ...
```

### Architecture

```markdown
# [Project] Architecture

## Overview
High-level description.

## Components
| Component | Responsibility |
|-----------|----------------|
| ... | ... |

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| ... | ... |
```

### Story

```markdown
# [STORY-ID] [Title]

## Goal
What are we trying to accomplish?

## Acceptance Criteria
- [ ] ...

## Context Scope
**Write:** ...
**Read:** ...
**Exclude:** ...

## Approach & Verification Plan
How to verify success (tests, manual steps).
```

### Session Summary

```markdown
# Session: [Topic]
**Date:** YYYY-MM-DD

## Overview
What was accomplished.

## Key Decisions
- ...

## Artifacts Modified
- ...

## Next Steps
- ...
```

## 12. Traina Project: System Architecture

### Production System: threading_ws

**Location:** `threading_ws/` - Complete ROS2 workspace (16 packages)

**Status:** Production-ready, self-contained, copied from backup 2025-02-05

**Key Capabilities:**
- HDR multi-layer imaging (ImageSetHdr)
- 3-model AI pipeline (segmentation + classification + stem detection)
- 3-class classification (clean/maybe/pit)
- 8 selectable detection algorithms
- C++ action-based orchestration (composite)
- PLC-based I/O (plc_eip)

**Model Files:** ~1.1GB in `threading_ws/src/cherry_detection/resource/` (excluded from git)

**Architecture Documentation:**
- [threading_ws INDEX](docs/core/architecture/threading_ws/INDEX.md) - System overview
- [cherry_interfaces](docs/core/architecture/threading_ws/cherry_interfaces/ARCHITECTURE.md) - Expanded interfaces
- [Migration Guide](docs/reference/MIGRATION_cherry_system_to_threading_ws.md) - cherry_system → threading_ws

### Legacy System: cherry_system (Archived)

**Location:** `cherry_system/` - Retained for reference

**Status:** Legacy, 2-model pipeline, simpler interfaces

### Experiment Tracking

**Current Phase:** Model optimization experiments for cherry pit detection

**Key Reference Documents:**
| Document | Purpose |
|----------|---------|
| `docs/reference/MODEL_EXPERIMENTS.md` | Complete log of all model experiments and results |
| `docs/reference/MIGRATION_cherry_system_to_threading_ws.md` | System migration guide |
| `docs/logs/` | Timestamped session summaries |
| `docs/stories/STORY-005-Progress-Review-And-Planning.md` | Next session: review + roadmap planning |

**Current Best Models:**
| Model | Accuracy | Size | Status | System |
|-------|----------|------|--------|--------|
| Production (ResNet50) | 92.99% | 90MB | Baseline | cherry_system |
| ResNet50 Unnormalized | 94.05% | 90MB | Best (experiment complete) | cherry_system |
| ResNet18 | 91.92% | 43MB | Complete (speed candidate) | cherry_system |
| **classification-2_26_2025-iter5.pt** | Unknown | 91MB | **Production** | **threading_ws** |

**Active Infrastructure:**
- Colab notebook: `training/notebooks/colab_optimization_experiments.ipynb` (skip-flag pattern)
- Threshold optimization: Ready to run (code in place, production model path documented)
- Model evaluation: Production model evaluation cell added to notebook
- **threading_ws:** Complete production workspace with 3-model pipeline

**Next Action:** STORY-005 (start next session with progress review)

## 13. Agent Guidelines

- **Search first.** Exhaust search tools before asking for paths.
- **Tool Selection.** Skills for vertical workflows, framework docs for APIs, external tools for navigation.
- **Prefer retrieval over training.** Consult `docs/` for framework specifics.
- **Read before writing.** Check Context Scope and Read files.
- **Update, don't accumulate.** Modify docs in place—git tracks history.
- **Compact proactively.** Don't wait for context overflow.
- **Stay in scope.** Don't modify files outside Write scope without approval.
- **Be concise.** Terse, technical communication.
