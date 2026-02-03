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
    3.  **Classification**: **ResNet50** (or MobileNet/EfficientNet) classifies each crop as `Clean`, `Pit`, `Maybe`, or `Side`.

### 2. Training Infrastructure (`training/`)
*   **Location**: Root `training/` directory.
*   **Workflow**:
    *   Data managed on Google Drive.
    *   Training runs on Google Colab Pro (due to GPU requirements).
    *   Scripts: `training/scripts/train.py`, `inspect_model.py`.
    *   **Unnormalized Training**: Models are explicitly trained on unnormalized data to match the production pipeline.

## Technical Debt & Known Issues
*   **Code Duplication**: `ai_detector.py` logic exists in both `cherry_detection` and `control_node`. The `cherry_detection` version is the canonical one, but verify which is actually imported at runtime.
*   **Model Loading**: There is a known configuration bug where weights might be loaded from `control_node/resource` instead of `cherry_detection/resource`.
*   **Denormal Values**: ResNet50 on CPU is sensitive to "denormal" float values which cause massive slowdowns. We use `fix_denormals.py` to sanitize weights.

## Discovery Links
*   **Code**: `src/cherry_system/cherry_detection/`
*   **Training Code**: `training/`
*   **Analysis**: [ResNet50 Analysis](./RESNET50_ANALYSIS.md)
*   **Skill**: `../../skills/benchmark-latency/`
