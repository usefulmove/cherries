# PyTorch Tutorial: Cherry Processing Machine - Section 1: Overview

## Two-Stage ML Pipeline

This cherry processing system uses a two-stage deep learning pipeline:

1. **Mask R-CNN** - Instance segmentation: Detects individual cherries in camera images
2. **ResNet50** - Classification: Determines if each detected cherry is clean or contains a pit

```
┌─────────────┐
│ Camera Image│
│  (CV2/NumPy)│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Mask R-CNN    │  ← Segment: Find all cherry instances
│  (torchvision)  │
└──────┬──────────┘
       │
       │  Bounding boxes + Masks
       ▼
┌─────────────────┐
│    ResNet50     │  ← Classify: Clean vs Pit
│  (torchvision)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Cherry List     │  (x, y, type, confidence)
│  (ROS2 Output)  │
└─────────────────┘
```

## File Structure

### Core Implementation
```
cherry_system/cherry_detection/cherry_detection/
├── ai_detector.py          # Main PyTorch implementation
└── resource/
    ├── cherry_segmentation.pt   # Mask R-CNN weights
    └── cherry_classification.pt # ResNet50 weights
```

### Integration (ROS2 Wrapper)
```
cherry_system/control_node/control_node/
├── ai_detector.py          # Duplicate for control node
└── detector.py            # Wrapper: ML → ROS2 messages
```

## What Each Model Does

### Mask R-CNN (Segmentation)
- **Input:** Full camera image (~2463x500 pixels)
- **Output:**
  - Bounding boxes: `[x1, y1, x2, y2]` for each cherry
  - Masks: Pixel-level segmentation (shape: `[N, 1, H, W]`)
  - Scores: Detection confidence
- **Purpose:** Locate all cherry instances in the conveyor image
- **Classes:** 2 classes (0: background, 1: cherry matter)

**Key code reference:** `ai_detector.py:30-75` - `get_instance_segmentation_model()`

```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=False,
    num_classes=2,
    min_size=800,
    max_size=2464,
    # ... additional parameters
)
```

### ResNet50 (Classification)
- **Input:** 128x128 crop of each detected cherry (masked by segmentation)
- **Output:**
  - Logits: Raw scores for each class
  - Softmax: Probability distribution `[prob_clean, prob_pit]`
- **Purpose:** Determine cherry quality (good vs. defective)
- **Classes:** 2 classes (0: clean, 1: pit)

**Key code reference:** `ai_detector.py:148-156`

```python
self.classifier = resnet50().to(self.device)
num_ftrs = self.classifier.fc.in_features
self.classifier.fc = torch.nn.Linear(num_ftrs, 2)  # Binary classification
self.classifier.load_state_dict(weights2)
self.classifier.eval()
```

## Model Weights

The system loads pre-trained weights from `.pt` files:

- **Segmentation:** `cherry_segmentation.pt` - ~260MB Mask R-CNN model
- **Classification:** `cherry_classification.pt` - ~90MB ResNet50 model

**Key code reference:** `ai_detector.py:139, 154`

```python
self.model.load_state_dict(weights)  # Mask R-CNN
self.classifier.load_state_dict(weights2)  # ResNet50
```

## Integration with ROS2

The detector is wrapped in a ROS2 service (`detector.py`):

1. Receives camera image via ROS2 topic/service
2. Runs PyTorch inference
3. Converts detections to `Cherry` messages with real-world coordinates
4. Returns cherry list for downstream actuation

**Key code reference:** `detector.py:143-183`

```python
def detect(self, image_color, logger):
    dets, kp_im_processed = self.detector.detect(image_color)
    dets = self.real_world(dets)  # Convert pixels → meters
    # ... create Cherry messages
    return cherries, kp_im_processed
```

## Next Section

**[02. Models](PYTORCH_02_MODELS.md)** - Model definition and loading patterns
