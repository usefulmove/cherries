# PyTorch Tutorial: Cherry Processing Machine - Section 8: Post-Processing

## Label Assignment

**Reference:** `ai_detector.py:376-383, 461-462`

```python
# Get probabilities from softmax
probs = torch.nn.functional.softmax(classifications, dim=1)
# probs shape: [N, 2] → [[p_clean, p_pit], ...]

# Get max probability and class
conf, classes = torch.max(probs, 1)
# conf: [N], classes: [N]

# Create boolean masks for each category
pit_mask = probs[:, 1].ge(0.75)     # pit_prob >= 75%
maybe_mask = probs[:, 1].ge(0.5)     # pit_prob >= 50%
clean_mask = probs[:, 0].ge(0.5)     # clean_prob >= 50%

# Set labels based on probability thresholds
# Order matters! Later where() calls override earlier ones
prediction['labels'] = torch.where(maybe_mask, 5, prediction['labels'])  # Maybe → 5
prediction['labels'] = torch.where(pit_mask, 2, prediction['labels'])    # Pit → 2
prediction['labels'] = torch.where(clean_mask, 1, prediction['labels'])  # Clean → 1
```

**Label codes:**
| Code | Meaning | Threshold |
|------|---------|------------|
| 0 | Background | N/A |
| 1 | Clean cherry | clean_prob ≥ 0.5 |
| 2 | Cherry with pit | pit_prob ≥ 0.75 |
| 3 | Edge detection | x < 170 or x > 2244 |
| 5 | Maybe pit | pit_prob ≥ 0.5 |

## Edge Detection

**Reference:** `ai_detector.py:461-462`

```python
# Mark cherries at conveyor edges as "side" (label 3)
prediction['labels'] = torch.where(
    prediction['boxes'][:, 0] < 170,  # Left edge: x1 < 170
    3, 
    prediction['labels']
)

prediction['labels'] = torch.where(
    prediction['boxes'][:, 2] > 2244,  # Right edge: x2 > 2244
    3, 
    prediction['labels']
)
```

## Filtering by Label

**Reference:** `ai_detector.py:474-477`

```python
# Create boolean masks for each label type
clean_mask = prediction['labels'].eq(1)  # Equal to 1
pit_mask = prediction['labels'].eq(2)    # Equal to 2
side_mask = prediction['labels'].eq(3)   # Equal to 3
maybe_mask = prediction['labels'].eq(5)   # Equal to 5
```

**Use masks for indexing:**
```python
# Get all clean cherries
clean_boxes = prediction['boxes'][clean_mask]

# Get all pits
pit_boxes = prediction['boxes'][pit_mask]
```

## Drawing Bounding Boxes

**Reference:** `ai_detector.py:480-483`

```python
import torchvision.utils
import torchvision.transforms as T

# Prepare image for drawing
img_for_labeling = (img_tensor * 255).type(torch.uint8)  # [0, 1] → [0, 255]

# Draw boxes for each category with different colors
img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_for_labeling,
    prediction['boxes'][clean_mask],  # Clean cherries
    colors='limegreen',
    width=2
)

img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_labeled_tensor,                # Accumulate on previous image
    prediction['boxes'][pit_mask],      # Pits
    colors='red',
    width=2
)

img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_labeled_tensor,
    prediction['boxes'][side_mask],     # Edge cherries
    colors='cyan',
    width=2
)

img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_labeled_tensor,
    prediction['boxes'][maybe_mask],    # Maybe pits
    colors='yellow',
    width=2
)

# Convert back to CV2 format
topil = T.ToPILImage()
img_labeled = topil(img_labeled_tensor)
img_labeled = np.array(img_labeled)
img_labeled = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)
```

**`draw_bounding_boxes` parameters:**
- `image`: Tensor of shape `(C, H, W)` with dtype `uint8`
- `boxes`: Tensor of shape `(N, 4)` with `[x1, y1, x2, y2]` format
- `labels`: Optional tensor of shape `(N,)` for label text
- `colors`: Color for boxes (string or tuple)
- `width`: Box line thickness

## Result Aggregation

**Reference:** `ai_detector.py:503-507`

```python
# Convert to NumPy arrays
boxes = prediction['boxes'].cpu().numpy()
labels = prediction['labels'].cpu().numpy()
confidence_probs = prediction['confidence_probs'].cpu().numpy()

# Create result dictionary
filtered = {
    'boxes': boxes,              # [N, 4] bounding boxes
    'confidences': confidence_probs,  # [N, 2] probabilities
    'labels': labels              # [N] label codes
}
```

## Converting to Real-World Coordinates

**Reference:** `detector.py:114-139`

```python
def real_world(self, detections):
    px_boxes = detections['boxes']
    real_boxes = []
    
    for bbox in px_boxes:
        # Convert pixel bbox to world coordinates
        xy1 = self.process_point((bbox[0], bbox[1]))  # Top-left
        xy2 = self.process_point((bbox[2], bbox[3]))  # Bottom-right
        
        real_boxes.append([xy1[0], xy1[1], xy2[0], xy2[1]])
    
    detections['real_boxes'] = real_boxes
    return detections

def process_point(self, pt):
    # Rotate and scale to meters
    pt_rotated = self.rotate(pt)
    pt_scaled = self.scale(pt_rotated)
    return pt_scaled
```

## Creating Cherry Messages

**Reference:** `detector.py:159-171`

```python
cherries = []

for i in range(len(dets['labels'])):
    xyxy = dets['real_boxes'][i]  # Real-world bbox
    confidence = dets['confidences'][i]
    cls = int(dets['labels'][i])
    
    cherry = Cherry()
    cherry.x = (xyxy[0] + xyxy[2]) / 2.0  # Center x
    cherry.y = (xyxy[1] + xyxy[3]) / 2.0  # Center y
    cherry.type = (cls).to_bytes(1, 'big')  # Label as bytes
    
    cherries.append(cherry)
```

## Complete Post-Processing Flow

```
Raw Classification Output
     │
     ▼ Softmax
probs [N, 2]
     │
     ▼ torch.max()
conf, classes
     │
     ▼ Boolean masks
clean_mask, pit_mask, etc.
     │
     ▼ torch.where()
final labels [N]
     │
     ├─→ Filter by label
     │   └─→ boxes[mask]
     │
     ├─→ Draw boxes
     │   └─→ draw_bounding_boxes()
     │
     └─→ Aggregate
         └─→ {boxes, confidences, labels}
```

## Next Section

**`PYTORCH_09_COMPLETE_PIPELINE.md`** - End-to-end walkthrough
