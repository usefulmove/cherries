---
name: Inference Pipeline
layer: AI/ML
impact_area: Accuracy, Classification, Training
---

# Inference Pipeline Layer

## Responsibility
The "Brain" of the inspection. Detects cherries in the image (Segmentation) and determines if they contain pits (Classification).

## Key Components

### 1. Detection Service (`cherry_system/cherry_detection/`)
*   **Node**: `cherry_detection`
*   **Service**: `DetectCherries`
*   **Pipeline**:
    1.  **Segmentation**: **Mask R-CNN** identifies cherry ROIs (Regions of Interest).
    2.  **Preprocessing**: Crops and resizes cherries. **Crucial:** Does *not* normalize pixel values (uses raw 0-255) to save CPU cycles.
    3.  **Classification**: **ResNet50** (or MobileNet/EfficientNet) classifies each crop into one of four categories based on confidence thresholds.

#### Classification Categories

The model outputs probabilities for two classes (clean vs pit), which are mapped to four operational categories:

| Label | Category | Threshold | Description | Visualization |
|:------|:---------|:----------|:------------|:--------------|
| 1 | **Clean** | clean prob ≥ 0.5 | Confidently no pit | Green bounding box / circle |
| 2 | **Pit** | pit prob ≥ 0.75 | Confidently has pit | Red bounding box / circle |
| 3 | **Side** | edge detection | Cherry at image edge | Cyan bounding box / circle |
| 5 | **Maybe** | 0.5 ≤ pit prob < 0.75 | Uncertain—requires manual review | **Yellow bounding box / circle** |

**Key Implementation Details:**
- **Locations array** (`ai_detector.py:246`): `['none', 'cherry_clean', 'cherry_pit', 'side', 'top/bot', 'maybe']`
- **Threshold logic** (`ai_detector.py:376-383`): 
  - `pit_mask = probs[:, 1].ge(.75)` → label 2
  - `maybe_mask = probs[:, 1].ge(.5)` → label 5 (if not pit)
  - `clean_mask = probs[:, 0].ge(.5)` → label 1
- **Safety Feature**: The "Maybe" category (label 5) creates a manual review pathway—uncertain predictions are highlighted in yellow on the belt projection for worker inspection rather than being automatically sorted.

### 2. Training Infrastructure (`training/`)
*   **Location**: Root `training/` directory.
*   **Workflow**:
    *   Data managed on Google Drive.
    *   Training runs on Google Colab Pro (due to GPU requirements).
    *   Scripts: `training/scripts/train.py`, `inspect_model.py`.
    *   **Unnormalized Training**: Models are explicitly trained on unnormalized data to match the production pipeline.

## Model Paths

| Model | Path | Description |
|-------|------|-------------|
| **Production (Canonical)** | `cherry_system/cherry_detection/resource/cherry_classification.pt` | Currently deployed model (92.99% accuracy) |
| **Production (Duplicate)** | `cherry_system/control_node/resource/cherry_classification.pt` | May be loaded due to config bug - see known issues |
| **Best Training** | `training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt` | Our best trained model (94.05% accuracy) |

**Note:** When deploying a new model, update the canonical path. The `control_node` copy exists due to a known bug and should eventually be removed.

## Technical Debt & Known Issues
*   **Code Duplication**: `ai_detector.py` logic exists in both `cherry_detection` and `control_node`. The `cherry_detection` version is the canonical one, but verify which is actually imported at runtime.
*   **Model Loading**: There is a known configuration bug where weights might be loaded from `control_node/resource` instead of `cherry_detection/resource`. Always verify which model is being loaded at runtime.
*   **Denormal Values**: ResNet50 on CPU is sensitive to "denormal" float values which cause massive slowdowns. We use `fix_denormals.py` to sanitize weights.

## Discovery Links
*   **Code**: `src/cherry_system/cherry_detection/`
*   **Training Code**: `training/`
*   **Analysis**: [ResNet50 Analysis](./RESNET50_ANALYSIS.md)
*   **Skill**: `../../skills/benchmark-latency/`
*   **Visualization**: See [Tracking & Orchestration](../tracking_orchestration/ARCHITECTURE.md) for projection system details
*   **Category Handling**: `cherry_system/cherry_detection/cherry_detection/ai_detector.py:246, 376-383` (classification logic)
