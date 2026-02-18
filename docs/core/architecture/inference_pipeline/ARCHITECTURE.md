---
name: Inference Pipeline
layer: AI/ML
impact_area: Accuracy, Classification, Training
---

# Inference Pipeline Layer

## Responsibility

The "Brain" of the inspection. Detects cherries in the image (Segmentation), determines if they contain pits (Classification), and identifies stem locations (Stem Detection).

## Architecture Overview

The current production system uses a **3-model pipeline** (algorithm v6/hdr_v1) consisting of:

1. **Segmentation Model** (Mask R-CNN): Detects cherry locations and generates masks
2. **Classification Model** (ResNet50): Classifies cherries into quality categories
3. **Stem Detection Model** (Faster R-CNN): Detects stem locations in the color image

## Current Production Pipeline (hdr_v1 / v6)

### Model Configuration

| Model | Architecture | Input | Output | Classes | File |
|-------|-------------|-------|--------|---------|------|
| **Segmentation** | Mask R-CNN ResNet50 FPN | Grayscale (500×2464) (Red channel stacked 3x) | Masks, boxes, scores | 2 (bg, cherry) | `seg_model_red_v1.pt` |
| **Classification** | ResNet50 | 128×128 crops | Quality scores | **3** (clean, maybe, pit) | `classification-2_26_2025-iter5.pt` |
| **Stem Detection** | Faster R-CNN ResNet50 FPN v2 | Color RGB (cropped to 500×2464 top strip) | Stem bounding boxes | 2 (bg, stem) | `stem_model_10_5_2024.pt` |

### Detection Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Grayscale Image│────▶│  Mask R-CNN      │────▶│ Cherry ROIs     │
│  (Red Channel)  │     │  Segmentation    │     │ + Masks         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                  │
         │ (Stacked 3x to fake RGB)                         ▼
         │                                         ┌──────────────────┐
         │                                         │ Crop & Resize    │
         │                                         │ 128×128 patches  │
         │                                         └──────────────────┘
         │                                                  │
         │                                                  ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Color Image    │────▶│ Faster R-CNN     │     │ ResNet50         │
│  (RGB aligned)  │     │ Stem Detection   │     │ Classification   │
└─────────────────┘     └──────────────────┘     └──────────────────┘
         │               (Cropped to top            (3-Class Output:
         │                strip: y=40-480)           Clean, Maybe, Pit)
         ▼                                                  │
┌─────────────────┐                              ┌──────────────────┐
│ Stem Boxes      │                              │ Class Labels     │
│ (Type 6)        │                              │ (1,2,3,5)        │
└─────────────────┘                              └──────────────────┘

                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ Final Detections │
                                                │ + Stem Markers   │
                                                └──────────────────┘
```

## Classification Categories

The classifier outputs probabilities for **three classes**, which are mapped to operational categories:

| Label | Category | Model Output | Threshold | Description | Visualization |
|:------|:---------|:-------------|:----------|:------------|:--------------|
| 1 | **Clean** | Class 0 | clean prob ≥ 0.5 | Confidently no pit | Lime green bounding box |
| 2 | **Pit** | Class 2 | pit class detected | Confidently has pit | Red bounding box |
| 3 | **Side** | Position-based | Edge detection | Cherry at image edge | Cyan bounding box |
| 5 | **Maybe** | Class 1 | maybe class detected | Uncertain—requires manual review | **Yellow bounding box** |
| **6** | **Stem** | Stem model | Score ≥ 0.75 | Stem detected | **Black bounding box** |

**Evolution Note:** Earlier versions (v1-v5) used a 2-class classifier (clean/pit) with threshold-based maybe detection. The current 3-class classifier (v6+) uses explicit class predictions for clean/maybe/pit.

### Training Methodology Assessment

**Status:** The 3-class production model was trained using a **two-stage approach** with documented concerns:

| Stage | Process | Outcome |
|-------|---------|---------|
| **Stage 1** | Train binary classifier (pit vs no_pit) | Binary decision boundary |
| **Stage 2** | Fine-tune on misclassified examples labeled "maybe" | 3-class output (clean/maybe/pit) |

**Key Concerns:**
1. **Safety Risk:** Stage 1 misses become permanent false negatives
2. **Architecture Issues:** "Maybe" class violates mutual exclusivity; defined by errors, not ground truth
3. **Catastrophic Forgetting:** High risk of eroding Stage 1 learning during Stage 2
4. **Threshold Redundancy:** 3-class output converted back to thresholds (0.5/0.75) in production
5. **Auditability:** Complex cascade difficult to validate and explain

**Recommended Alternatives:**
- **Current 3-class explicit** (keep): Manually label "maybe" examples, train 3-class from scratch
- **Enhanced 2-class**: Binary classifier with calibrated confidence tiers (pit≥0.85 auto-reject, 0.65-0.85 review, <0.65 accept)
- **Ensemble methods**: Multiple models; disagreement indicates uncertainty

See [Training Methodology](/docs/reference/TRAINING_METHODOLOGY.md) for full assessment.

**Key Implementation Details:**
- **Locations array** (`ai_detector3.py:346`): `['none', 'cherry_clean', 'cherry_pit', 'side', 'top/bot', 'maybe']`
- **Classification logic** (`ai_detector3.py:616-625`): Uses `classes.eq()` for 3-class decisions
- **Stem integration** (`detector_node.py:473-485`): Stems assigned `type = 6` in Cherry messages

## Algorithm Versions

The system supports 8 algorithm configurations via dynamic parameter switching:

| Version | Algorithm Name | Detector Class | Models | Stem Support | Status |
|---------|---------------|----------------|--------|--------------|--------|
| v1 | fasterRCNN-Mask_ResNet50_V1 | ai_detector_class | 2-model | No | Legacy |
| v2 | fasterRCNN-NoMask_ResNet50_6-12-2023 | ai_detector_class_2 | 2-model | No | Legacy |
| v3 | newlight-mask-12-15-2023 | ai_detector_class | 2-model | No | Legacy |
| v4 | newlights-nomask-12-15-2023 | ai_detector_class_2 | 2-model | No | Legacy |
| v5 | NMS-nomask-1-3-2024 | ai_detector_class_2 | 2-model | No | Legacy |
| **v6** | **hdr_v1** | **ai_detector_class_3** | **3-model** | **Yes** | **DEFAULT** |
| v7 | hdr_v2 | ai_detector_class_3 | 3-model, no stem | Partial | Alternative |
| v8 | vote_v1 | ai_detector_class_4 | Multi-model ensemble | No | Experimental |

**Current Default:** `hdr_v1` (v6) is loaded on startup (`detector_node.py:75`).

## Model Paths

### Production Models (Current)

| Model | Path | Description |
|-------|------|-------------|
| **Segmentation** | `cherry_detection/resource/seg_model_red_v1.pt` | Cherry detection (Mask R-CNN) |
| **Classification** | `cherry_detection/resource/classification-2_26_2025-iter5.pt` | 3-class quality classifier |
| **Stem Detection** | `cherry_detection/resource/stem_model_10_5_2024.pt` | Stem location detector (166 MB) |

### Legacy Models

| Model | Path | Description |
|-------|------|-------------|
| **Original Classification** | `cherry_detection/resource/cherry_classification.pt` | 2-class baseline (92.99% accuracy) |
| **Best Training** | `training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt` | 94.05% accuracy |

**Note:** The original `cherry_classification.pt` (2-class) remains for backward compatibility with v1-v5 algorithms.

## Stem Detection Details

See [STEM_DETECTION.md](./STEM_DETECTION.md) for comprehensive stem detection documentation.

### Quick Overview

- **Architecture:** Faster R-CNN ResNet50 FPN v2
- **Input:** Aligned color image (3-channel RGB)
- **Output:** Stem bounding boxes with confidence scores
- **Threshold:** Score ≥ 0.75 for detection
- **Spatial Filter:** Focus on center region (not belt edges)
- **Integration:** Stems marked as Type 6 in Cherry messages
- **Visualization:** Black bounding boxes on processed images

### Current Status

The stem model is **loaded and executed** in production but its practical sorting impact is **under investigation**. See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md).

## Training Infrastructure

*   **Location:** Root `training/` directory.
*   **Workflow:**
    *   Data managed on Google Drive.
    *   Training runs on Google Colab Pro (due to GPU requirements).
    *   Scripts: `training/scripts/train.py`, `inspect_model.py`.
    *   **Unnormalized Training:** Models are explicitly trained on unnormalized data (0-255 range) to match the production pipeline.

### Training Data

See [Training Data Reference](../../reference/training-data.md) for dataset details.

**Key Datasets:**
- Classification: GitHub repo + collected images (inherently 2-class: clean/pit)
- Segmentation: COCO/VOC annotations
- **Stems:** `/media/dedmonds/Extreme SSD/traina cherry line/Pictures/hdr/20240923 stems/` (~570 samples)

**Note:** Training data contains only clean/pit labels. The "maybe" class was synthetically created from Stage 1 misclassifications, not manually labeled. See [Training Methodology](../../reference/TRAINING_METHODOLOGY.md).

## Technical Debt & Known Issues

*   **Code Duplication:** Multiple `ai_detector` versions (ai_detector.py through ai_detector4.py) exist. The `ai_detector3.py` version is canonical for v6+ operations.
*   **Model Loading:** There is a known configuration bug where weights might be loaded from `control_node/resource` instead of `cherry_detection/resource`. Always verify which model is being loaded at runtime.
*   **Denormal Values:** ResNet50 on CPU is sensitive to "denormal" float values which cause massive slowdowns. We use `fix_denormals.py` to sanitize weights.
*   **Stem Detection Purpose:** The practical use of stem detections in the sorting logic is unclear. See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md).
*   **Training Methodology Concerns:** The two-stage training approach (binary → 3-class via error fine-tuning) has documented safety and architectural risks. See [Training Methodology](../../reference/TRAINING_METHODOLOGY.md).

## Discovery Links

*   **Code:** `threading_ws/src/cherry_detection/`
*   **Backup Code:** `/media/dedmonds/Extreme SSD/traina cherry line/threading_ws/src/cherry_detection/`
*   **Training Code:** `training/`
*   **Analysis:** [ResNet50 Analysis](./RESNET50_ANALYSIS.md)
*   **Stem Detection:** [STEM_DETECTION.md](./STEM_DETECTION.md)
*   **Skill:** `../../skills/benchmark-latency/`
*   **Visualization:** See [Tracking & Orchestration](../tracking_orchestration/ARCHITECTURE.md) for projection system details
*   **Category Handling:** `threading_ws/src/cherry_detection/cherry_detection/ai_detector3.py:346, 616-625` (3-class classification logic)
*   **Stem Integration:** `threading_ws/src/cherry_detection/cherry_detection/detector_node.py:473-485` (stem message handling)
