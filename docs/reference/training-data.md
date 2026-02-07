# Training Data Reference

**Sources:** GitHub Repository, Collected Production Data, External Drive (backup)

---

## Classification Training Data

### Primary Dataset

**Source:** `https://github.com/weshavener/cherry_classification`

- **Collected:** November 2, 2022
- **Structure:** Train/validation split
- **Format:** Image files with directory-based labels
- **Access:** Cloned in training notebooks

### Production Data Collection

Located on external drive at `/media/dedmonds/Extreme SSD/traina cherry line/Pictures/hdr/`:

#### Cherry Type Categories

| Category | Directory | Count | Description |
|:---------|:----------|:------|:------------|
| Natural Pits | `natural_pits/` | 258 | Naturally occurring pits |
| Natural Clean | `natural_clean/` | 102 | Clean natural cherries |
| Organic Pits | `organic_pits/` | 189 | Organic variety with pits |
| Organic Clean | `organic_clean/` | 127 | Organic variety clean |
| Sulfur Pits | `sulfur_pits/` | Unknown | Sulfur-treated with pits |
| Sulfur Clean | `sulfur_clean/` | Unknown | Sulfur-treated clean |
| Recent Clean | `20240611_clean/` | 1,700+ | Recent collection (June 2024) |
| Error Cases | `20240423 missed pits/` | Unknown | Misclassification examples |

**Total:** 11.8 GB, 11,800+ images (estimated)

**Note:** Images are manually collected and categorized, NOT auto-labeled.

---

## Stem Training Data

### Stem Detection Dataset (2024)

**Location:** `/media/dedmonds/Extreme SSD/traina cherry line/Pictures/hdr/20240923 stems/`

**Metadata:**
- **Collected:** September 23, 2024
- **Model Trained:** October 5, 2024 (`stem_model_10_5_2024.pt`)
- **Samples:** ~570 timestamped directories
- **Format:** Raw camera captures (timestamped subdirectories)

**Structure:**
```
20240923 stems/
├── 20240923T171148898032/
├── 20240923T171149945551/
├── 20240923T171326143876/
├── ... (~570 total directories)
└── 20240923T.../
```

Each directory contains individual cherry images with stems present.

### Historical Stem Data

| Directory | Date | Contents |
|:----------|:-----|:---------|
| `20240423 bad small stems and misc/` | April 23, 2024 | Small stems, edge cases |
| `20240423 larges stems and misc bad/` | April 23, 2024 | Large stems, problem cases |

These earlier collections may have been used for validation or experimental training.

### Training Unknowns

See [Open Questions: Stem Detection](../reference/open-questions-stem-detection.md):
- Annotation format (COCO, YOLO, custom?)
- Label schema (bounding boxes, segmentation masks?)
- Training script location
- Validation methodology
- Performance metrics

---

## Segmentation Training Data

### YOLOv7 Format Dataset

**Location:** `/media/dedmonds/Extreme SSD/traina cherry line/pytorch/cherry_data/`

**Statistics:**
- **Total:** 566 annotated images
- **Train:** 394 images
- **Validation:** 113 images
- **Test:** 59 images
- **Size:** 1.2 GB

**Format:** COCO/VOC/YOLOv7 compatible annotations

**Classes:**
- `null_val`
- `with_pit`
- `clean`

### Roboflow Exports

Multiple dataset versions exported from Roboflow:
- `Cherry Inspection.v3i.*` (various formats)
- `Cherry inspection -2.v3i.*`

---

## Data Management Workflow

### Collection Pipeline

1. **Production Capture:** Images captured during sorting operations
2. **Manual Review:** Operators identify interesting/problematic cases
3. **Categorization:** Sorted into type directories (pits/clean/stems)
4. **Backup:** Synced to external drive (`/media/dedmonds/Extreme SSD/`)
5. **Training:** Selected batches uploaded to Google Drive for Colab training

### Training Infrastructure

- **Primary:** Google Colab Pro (GPU required)
- **Storage:** Google Drive for dataset hosting
- **Scripts:** `training/scripts/train.py`, `inspect_model.py`
- **Unnormalized Training:** All models trained on raw 0-255 pixel values

---

## Model-Specific Data Requirements

### Classification Model (ResNet50)

**Input:** 128×128 pixel crops
**Classes:** 
- v1-v5: 2-class (clean, pit)
- v6+: 3-class (clean, maybe, pit)

**Important:** Training data contains only clean/pit labels. The "maybe" class was synthetically created via two-stage training (Stage 1: binary classifier, Stage 2: fine-tune on misclassifications as "maybe"). This approach has documented safety and architectural concerns. See [Training Methodology](./TRAINING_METHODOLOGY.md).

**Preprocessing:**
- Center crop to 128×128
- No normalization (0-255 range)
- Mask applied to isolate cherry from background

### Segmentation Model (Mask R-CNN)

**Input:** Full resolution (500×2464) grayscale
**Output:** Instance masks + bounding boxes
**Annotation:** COCO-style segmentation masks

### Stem Detection Model (Faster R-CNN)

**Input:** Aligned color image (3×500×2464)
**Output:** Bounding boxes only (no masks)
**Annotation:** Object detection format (likely COCO or YOLO)

**Training Data:**
- ~570 positive samples (stems present)
- Unknown negative samples (no stems)
- Spatial focus: Center region of belt

---

## Data Quality Notes

### Challenges

1. **Class Imbalance:** More clean cherries than pits in production
2. **Edge Cases:** Stems, side positions, occlusions
3. **Lighting Variation:** Different camera exposures, HDR processing
4. **Manual Labeling:** Time-intensive, subject to human error

### Recent Collections

- **June 2024:** `20240611_clean/` - Large batch of clean cherries (1,700+ images)
- **September 2024:** `20240923 stems/` - Dedicated stem collection for 3rd model
- **Ongoing:** Error cases captured from production misclassifications

---

## Related Documentation

- [Stem Detection](../core/architecture/inference_pipeline/STEM_DETECTION.md)
- [Open Questions: Stem Detection](../reference/open-questions-stem-detection.md)
- [Inference Pipeline](../core/architecture/inference_pipeline/ARCHITECTURE.md)
- [Training Infrastructure](../../training/)
