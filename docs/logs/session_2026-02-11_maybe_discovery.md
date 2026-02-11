# Session: Maybe Class Discovery & Pipeline Confirmation
**Date:** 2026-02-11

## Overview
Analyzed the original training notebooks (`temp-old-training-docs/`) and compared them with the live production code (`threading_ws`) to resolve ambiguities about the "Maybe" class, segmentation inputs, and stem detection.

## Key Discoveries

### 1. The "Maybe" Class is Active Learning
The "Maybe" class isn't just a low-confidence threshold. It's the result of an iterative **Active Learning** workflow:
1.  Train 2-class model (Clean/Pit).
2.  Run inference.
3.  Filter low-confidence predictions into a "Maybe" folder using a script (`sort_for_maybe`).
4.  Re-train with 3 classes (Clean/Maybe/Pit).

This confirms the "Safety Valve" pattern: the model explicitly learns to recognize and reject ambiguous inputs (glare, stems, blur).

### 2. Segmentation "Red Channel" Trick
The segmentation model uses a specific preprocessing trick:
-   **Input:** Red channel only.
-   **Method:** The Red channel is stacked 3 times (`img[:,:,0]=R, img[:,:,1]=R, ...`) to simulate a 3-channel input for the standard ResNet backbone.

### 3. Stem Detection Recipe
We found the exact training recipe for the Stem Detector in `stems_3channel.ipynb`:
-   **Model:** Faster R-CNN ResNet50 FPN v2.
-   **Crucial Transform:** `Crop(0, 40, 2464, 480)`. The model is trained on and sees **only the bottom strip** of the image.

### 4. Unnormalized Training Confirmed
The original notebooks explicitly comment out normalization. This validates our [2026-01-30] decision to switch to unnormalized training to match the production pipeline.

## Decisions
1.  **Keep the "Maybe" Workflow:** It's a valid industrial pattern. We will modernize it (using `fiftyone` instead of manual file copying) rather than trying to engineer it away.
2.  **Stick to Unnormalized:** All future training will use the "unnormalized" data pipeline.
3.  **Document Everything:** Updated `ARCHITECTURE.md` and `LESSONS.md` with these findings to prevent future confusion.

## Next Steps
-   **Discuss:** "Maybe" class handling workflow (modernization vs legacy script).
-   **Implement:** A modernized `sort_for_maybe.py` using `fiftyone` or a simple Python script to automate the active learning loop.
