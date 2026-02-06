# Cherry Processing System: Open Questions - Stem Detection

**Date:** 2025-02-05  
**Status:** Active Investigation  
**Related:** Inference Pipeline Architecture, ai_detector3.py, stem_model_10_5_2024.pt

---

## Context

The production system (as of February 2025) uses a **3-model pipeline** (algorithm v6/hdr_v1) that includes a stem detection model (`stem_model_10_5_2024.pt`, 166 MB, October 5, 2024). While the model is loaded and executed, its practical purpose and integration into the sorting logic are unclear.

## Open Questions

### 1. What is the practical purpose of stem detection?

**Current Behavior:**
- The stem detection model runs on every image
- Detected stems are assigned `type = 6` in the Cherry message
- Stems are visualized with **black bounding boxes** on the projection system
- Stems appear to be tracked alongside cherries in the detection pipeline

**Unknowns:**
- Does stem presence trigger a reject decision?
- Are cherries with stems routed differently?
- Is stem data used for quality grading (e.g., premium vs standard)?
- Is this purely for data collection/statistical purposes?

**Technical Context:**
```python
# From detector_node.py:475-485
if (len(stems['labels']) > 0):
    stems = self.real_world(stems)
    stem_cls = 6  # Type 6 = Stem
    for i in range(len(stems['labels'])):
        xyxy = stems['real_boxes'][i]
        cherry = Cherry()
        cherry.x = (xyxy[0] + xyxy[2]) / 2.0
        cherry.y = (xyxy[1] + xyxy[3]) / 2.0
        cherry.type = (stem_cls).to_bytes(1, 'big')
        cherries.append(cherry)
```

### 2. How do stem detections interact with the robot/pneumatic system?

**Current Integration:**
- Stems are added to the CherryArray message published to the tracking system
- Type 6 labels are transmitted through the detection pipeline
- Robot controller receives stem locations alongside cherry locations

**Questions:**
- Does the robot act on stem detections (e.g., separate collection)?
- Are stems factored into the sorting decision for the associated cherry?
- Is there a separate actuator for stem removal?

### 3. What are the performance implications?

**Measured Latency:**
- Stem detection adds inference time to each image cycle
- Model architecture: Faster R-CNN ResNet50 FPN v2 (computationally expensive)
- Actual latency impact unknown

**Questions:**
- What is the latency cost of the stem model vs the 2-model pipeline?
- Could the stem model be run asynchronously or on a subset of images?
- Would disabling stem detection improve throughput for non-stem-critical operations?

### 4. Is there training data for the stem model?

**Evidence:**
- `/media/dedmonds/Extreme SSD/traina cherry line/Pictures/hdr/20240923 stems/` exists
- ~570 timestamped directories containing stem images (September 23, 2024)
- Date correlates with model creation date (October 5, 2024)

**Questions:**
- Was the stem model trained on this data?
- What is the label format (COCO, YOLO, custom)?
- Is the training code available?
- How was the model validated?

### 5. Why does the 3-class classifier co-exist with stem detection?

**Observation:**
- The hdr_v1 algorithm uses a 3-class classifier (clean, maybe, pit)
- This appears to be a different approach from the original 2-class system
- Stem detection runs in parallel with the 3-class classification

**Questions:**
- Is the 3-class classifier specifically designed for stem-affected cherries?
- Does stem presence influence the maybe/pit decision threshold?
- Was the classifier retrained to handle stem-occluded cherries better?

### 6. What is the fallback behavior if stem detection fails?

**Questions:**
- Is there error handling for stem model inference failures?
- Does the system gracefully degrade to 2-model operation?
- Are there health checks for the stem model?

## Research Needed

1. **Interview system operators** about stem detection in practice
2. **Review robot controller code** to see how type 6 (stem) is handled
3. **Analyze production logs** to see stem detection frequency
4. **Check sorting outcomes** for cherries with detected stems
5. **Benchmark latency** with and without stem model

## Related Documents

- [Inference Pipeline Architecture](../../core/architecture/inference_pipeline/ARCHITECTURE.md)
- [Stem Detection Details](../../core/architecture/inference_pipeline/STEM_DETECTION.md)
- [Training Data](../../training-data.md)

## Status History

- **2025-02-05:** Questions documented following discovery of stem_model_10_5_2024.pt on backup drive
