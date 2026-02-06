---
name: Stem Detection
component: Inference Pipeline
impact_area: Cherry Quality, Stem Identification
---

# Stem Detection System

## Overview

The stem detection system is the third component of the production inference pipeline (algorithm v6/hdr_v1). It uses a Faster R-CNN model to identify stem locations in cherry images, operating in parallel with the segmentation and classification models.

**Status:** The model is loaded and executed in production, but its practical impact on sorting decisions is under investigation. See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md).

## Model Specification

| Attribute | Value |
|:----------|:------|
| **File** | `stem_model_10_5_2024.pt` |
| **Size** | 166 MB |
| **Date** | October 5, 2024 |
| **Architecture** | Faster R-CNN ResNet50 FPN v2 |
| **Classes** | 2 (background, stem) |
| **Detector Class** | `ai_detector_class_3` (ai_detector3.py) |
| **Active In** | Algorithm v6 (hdr_v1) - current default |

## Architecture Details

### Model Configuration

```python
# ai_detector3.py:206-211
def get_stem_model(self, stem_weights):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    m.load_state_dict(stem_weights)
    return m
```

**Key Differences from Segmentation Model:**
- Uses **Faster R-CNN** (object detection) instead of Mask R-CNN (instance segmentation)
- No mask prediction head - only bounding box regression
- Same ResNet50 FPN backbone for feature extraction
- Optimized for stem-specific features (color, shape, texture)

### Input Processing

The stem model operates on the **aligned color image**:

1. **Image Source:** Multi-channel composite from bot1, bot2, and top2 cameras
2. **Alignment:** Images are offset-corrected based on encoder counts
3. **Format:** 3-channel RGB tensor (3 × 500 × 2464)
4. **Normalization:** Raw pixel values (0-255) - no normalization for speed

```python
# ai_detector3.py:474-477 (get_images method)
im_color = torch.zeros(3, 500, 2464)
im_color[0, :, :] = self.get_offset_image(images[0], 0)  # bot1
im_color[1, :, :] = self.get_offset_image(images[1], (counts[1] - counts[0]))  # bot2
im_color[2, :, :] = self.get_offset_image(images[2], (counts[2] - counts[0]))  # top2
```

## Detection Pipeline

### Step 1: Model Inference

```python
# ai_detector3.py:738-741
stem_start = time.time()
stem_prediction = self.detect_stems(img_color)
stem_end = time.time()
print(f'predict stems; {stem_end - stem_start}')
```

### Step 2: Spatial Filtering

Stems are filtered to focus on the center region of the belt:

```python
# ai_detector3.py:673-678
pred_mask = torch.logical_and(boxes[:, 1].ge(64), boxes[:, 3].le(446))    # Y range
pred_mask = torch.logical_and(pred_mask, boxes[:, 0].ge(125))             # X min
pred_mask = torch.logical_and(pred_mask, boxes[:, 2].le(2340))           # X max
pred_mask = torch.logical_and(pred_mask, scores.ge(0.75))                # Confidence
```

**Spatial Constraints:**
- X: 125 to 2340 pixels (avoid belt edges)
- Y: 64 to 446 pixels (vertical focus region)
- Score: ≥ 0.75 (high confidence threshold)

### Step 3: Output Format

The model returns a dictionary with:
- `boxes`: Bounding box coordinates [x1, y1, x2, y2]
- `scores`: Confidence scores (0.0 to 1.0)
- `labels`: Class labels (1 = stem)

## Integration with Detection Pipeline

### Data Flow

```
Color Image (3-channel)
        │
        ▼
┌───────────────────────┐
│ Faster R-CNN          │
│ Stem Detection        │
└───────────────────────┘
        │
        ▼
Stem Predictions (boxes, scores)
        │
        ▼
┌───────────────────────┐
│ Spatial Filtering     │
│ (center region focus) │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Message Generation    │
│ (type = 6)            │
└───────────────────────┘
        │
        ▼
CherryArray Message (published)
```

### Cherry Message Integration

Stems are converted to Cherry messages with a special type code:

```python
# detector_node.py:473-485
if (len(stems['labels']) > 0):
    stems = self.real_world(stems)  # Convert to real-world coordinates
    stem_cls = 6  # Type 6 = Stem
    for i in range(len(stems['labels'])):
        xyxy = stems['real_boxes'][i]
        cherry = Cherry()
        cherry.x = (xyxy[0] + xyxy[2]) / 2.0
        cherry.y = (xyxy[1] + xyxy[3]) / 2.0
        cherry.type = (stem_cls).to_bytes(1, 'big')
        cherries.append(cherry)
```

**Type Code Reference:**
- Type 1: Clean cherry
- Type 2: Pit detected
- Type 3: Side position (edge of belt)
- Type 5: Maybe (uncertain)
- **Type 6: Stem** ← Stem detection output

## Visualization

Stems are visualized with **black bounding boxes** on the processed output image:

```python
# ai_detector3.py:800
img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_labeled_tensor, 
    stem_prediction['boxes'], 
    colors='black', 
    width=2
)
```

This allows operators to visually confirm stem detection alongside cherry classifications (green/red/cyan/yellow).

## Training Data

### Dataset Location

**Path:** `/media/dedmonds/Extreme SSD/traina cherry line/Pictures/hdr/20240923 stems/`

**Contents:**
- ~570 timestamped directories
- Collected: September 23, 2024
- Model trained: October 5, 2024 (12-day gap suggests annotation/training period)

### Data Structure

Each timestamped directory contains:
- Individual cherry images with stems
- Likely annotated for object detection training
- Format appears to be raw camera captures

**Related Data:**
- `20240423 bad small stems and misc/` (April 2024 collection)
- `20240423 larges stems and misc bad/` (April 2024 collection)
- These may have been used for validation or earlier experiments

### Training Unknowns

See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md):
- Annotation format (COCO, YOLO, custom?)
- Training script location
- Validation methodology
- Performance metrics on stem dataset

## Performance Characteristics

### Computational Cost

- **Architecture:** Faster R-CNN ResNet50 FPN v2 (heavier than Mask R-CNN)
- **Inference Time:** Measured in logs but no baseline comparison available
- **GPU Memory:** Loads on CUDA if available (ai_detector3.py:183)

### Latency Impact

The stem model adds inference overhead to every image:

```python
# Timing instrumentation in ai_detector3.py
stem_start = time.time()
stem_prediction = self.detect_stems(img_color)
stem_end = time.time()
print(f'predict stems; {stem_end - stem_start}')
```

**Open Question:** What is the latency delta between 2-model and 3-model pipelines? See [benchmark-latency skill](../../skills/benchmark-latency/).

## Algorithm Version History

| Version | Date | Stem Support | Notes |
|:--------|:-----|:-------------|:------|
| v1-v5 | 2022-2024 | No | 2-model pipeline (seg + clf) |
| **v6 (hdr_v1)** | **Oct 2024** | **Yes** | **Current default - 3-model** |
| v7 (hdr_v2) | Oct 2024 | Partial | 3-model but stem disabled |
| v8 (vote_v1) | Jun 2024 | No | Multi-model ensemble |

## Comparison with Other Models

| Aspect | Segmentation | Classification | Stem Detection |
|:-------|:-------------|:---------------|:---------------|
| **Architecture** | Mask R-CNN | ResNet50 | Faster R-CNN |
| **Input** | Grayscale | 128×128 crops | Color (RGB) |
| **Output** | Masks + boxes | Class scores | Boxes only |
| **Classes** | 2 (bg, cherry) | 3 (clean/maybe/pit) | 2 (bg, stem) |
| **Purpose** | Find cherries | Quality assessment | Stem location |
| **Default** | All versions | All versions | v6+ only |

## Current Status and Open Questions

### What Works

- Model loads successfully
- Inference executes on every image
- Spatial filtering focuses on center region
- Output integrated into CherryArray messages
- Visualization displays stems in black

### What is Unknown

See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md) for full details:

1. **Purpose:** Does stem detection trigger sorting decisions?
2. **Robot Integration:** How does the robot controller handle Type 6 (stem) messages?
3. **Performance:** What is the latency cost vs 2-model operation?
4. **Training:** Where are the training scripts and annotations?
5. **Evolution:** How does stem detection relate to the 3-class classifier?

### Recommendations

1. **Investigate sorting logic:** Check robot controller code for Type 6 handling
2. **Benchmark latency:** Compare v6 (with stem) vs v7 (without stem)
3. **Review training data:** Examine annotation format in 20240923 stems/
4. **Interview operators:** Understand if stems are currently being handled differently

## Code References

- **Model Loading:** `ai_detector3.py:181-183`
- **Model Definition:** `ai_detector3.py:206-211`
- **Detection Method:** `ai_detector3.py:661-684`
- **Integration:** `ai_detector3.py:738-741`
- **Visualization:** `ai_detector3.py:800`
- **Message Creation:** `detector_node.py:473-485`
- **Algorithm Loading:** `detector_node.py:264-284` (load_v6)

## Related Documents

- [Inference Pipeline Architecture](./ARCHITECTURE.md)
- [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md)
- [Training Data](../../reference/training-data.md)
- [System Overview](../system_overview/ARCHITECTURE.md)
