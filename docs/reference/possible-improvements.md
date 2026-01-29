## Understanding the ML Pipeline

The system uses a **two-stage approach**:

1. **Instance Segmentation** (`ai_detector.py:30-75`): Mask R-CNN with ResNet50-FPN backbone
   - Detects cherry regions in images
   - Configurable at `get_instance_segmentation_model()` - 2 classes (background + cherry matter)
   - Key tunable parameters: `box_score_thresh=0.5`, `box_nms_thresh=0.5`, `rpn_nms_thresh=0.7`

2. **Classification** (`ai_detector.py:148-156`): ResNet50 classifier
   - Classifies each detected region as clean vs pit
   - Uses 128x128 cropped images from segmented masks
   - Threshold logic at lines 376-383: 
     - `pit_mask`: prob ≥ 0.75 → pit (label 2)
     - `maybe_mask`: prob ≥ 0.5 → maybe (label 5)  
     - `clean_mask`: prob ≥ 0.5 → clean (label 1)

---

## Plan to Explore Alternate Configurations

### Phase 1: Document Current Performance (2-3 days)
1. **Create a test dataset** - capture/representative images with ground truth labels
2. **Establish baseline metrics** - accuracy, precision, recall, F1 for clean/pit/maybe categories
3. **Profile inference timing** - identify bottlenecks (segmentation vs classification)

### Phase 2: Tune Detection Parameters (3-5 days)
1. **Adjust Mask R-CNN thresholds**:
   - `box_score_thresh` (0.5): Higher = fewer false positives, lower = more misses
   - `box_nms_thresh` (0.5): Lower = fewer overlapping detections
   - `rpn_fg_iou_thresh` / `rpn_bg_iou_thresh` (0.7/0.3): Affects region proposal quality
   
2. **Experiment with mask processing**:
   - Kernel size for dilation/erosion (currently 5x5)
   - Output image size (currently 128x128)

### Phase 3: Tune Classification Parameters (3-5 days)
1. **Adjust probability thresholds**:
   - `pit_mask` threshold (0.75): Higher = more conservative pit detection
   - `clean_mask` threshold (0.5): Lower = more aggressive clean classification
   - Consider adding a "reject" class for low-confidence predictions

2. **Data augmentation**: Test if more aggressive preprocessing improves robustness

### Phase 4: Advanced Options (1-2 weeks)
1. **Model architecture experiments**:
   - Try different backbones (ResNet34, EfficientNet)
   - Adjust FPN layers or RPN settings
2. **Ensemble methods**: Combine with alternative models
3. **Fine-tuning**: Retrain on your specific cherry dataset

---

## Key Files to Modify

| File | Purpose | Lines |
|------|---------|-------|
| `cherry_detection/cherry_detection/ai_detector.py` | Model config & inference | 30-75, 376-383 |
| `cherry_detection/cherry_detection/detector.py` | ROS2 wrapper, coordinate transforms | 143-180 |
| Model weights | `resource/cherry_segmentation.pt` (168MB), `resource/cherry_classification.pt` (90MB) |

---

## Questions Before Proceeding

1. **What classification accuracy are you targeting?** (e.g., 95% clean detection, <5% false pits)
2. **Do you have labeled training data** for potential retraining, or only want config tuning?
3. **What's the performance constraint?** (inference time requirements)
4. **Should I proceed with Phase 1** (baseline measurement setup) first?
