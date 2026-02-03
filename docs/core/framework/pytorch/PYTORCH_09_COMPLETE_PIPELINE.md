# PyTorch Tutorial: Cherry Processing Machine - Section 9: Complete Pipeline

## End-to-End Detection Flow

This walkthrough traces a single image through the entire ML pipeline.

**Reference:** `ai_detector.py:409-516`, `detector.py:143-183`

```python
import cv2
import torch
from cherry_detection.ai_detector import ai_detector_class
from detector import Detector

# ============================================
# 1. INITIALIZATION
# ============================================
# Load weights
weights = torch.load('cherry_segmentation.pt')
weights2 = torch.load('cherry_classification.pt')

# Create detector
my_detector = ai_detector_class(weights, weights2)
# Device: cuda
# Model: Mask R-CNN (seg), ResNet50 (class)
# Mode: eval()

# ============================================
# 2. INPUT IMAGE
# ============================================
# Raw camera image
img_cv2 = cv2.imread('camera_image.png')  # Shape: (H, W, 3) BGR uint8 [0, 255]
# Shape: (780, 2464, 3)

# ============================================
# 3. PREPROCESSING
# ============================================
# Crop to ROI
img_cv2 = img_cv2[200:700, 0:2463]  # Shape: (500, 2463, 3)

# BGR → RGB
img_pil = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

# PIL → Tensor (CHW, float [0, 1])
img_tensor = pil_to_tensor_gpu(img_pil, my_detector.device)
# Shape: (3, 500, 2463) on cuda

# ============================================
# 4. INSTANCE SEGMENTATION
# ============================================
with torch.no_grad():
    predictions = my_detector.model(img_tensor.unsqueeze(0))
    # Input: (1, 3, 500, 2463)
    
# Extract single image result
prediction = predictions[0]

# Output dictionary:
# - 'boxes': (N, 4)  [[x1, y1, x2, y2], ...]
# - 'labels': (N,)     [1, 1, 1, ...]  (all cherries)
# - 'scores': (N,)     [0.95, 0.92, 0.88, ...]
# - 'masks':  (N, 1, 500, 2463)

# Example: Found 12 cherries
N = len(prediction['masks'])  # N = 12

# ============================================
# 5. CLASSIFICATION PREPARATION
# ============================================
size_masks = prediction['masks'].size()  # torch.Size([12, 1, 500, 2463])

# Pre-allocate batch tensor
cl_imgs = torch.zeros(12, 3, 128, 128, device='cuda')

# Create kernel for mask processing
kernel = np.ones((5, 5), dtype=np.float32)
kernel_tensor = torch.tensor(
    np.expand_dims(np.expand_dims(kernel, 0), 0),
    device='cuda'
)  # Shape: (1, 1, 5, 5)

pad_im_transform = T.transforms.CenterCrop(128)

# ============================================
# 6. CHERRY CROP BATCHING
# ============================================
for index, mask in enumerate(masks):
    # Get bounding box
    bbox = boxes[index].type(torch.int)  # [x1, y1, x2, y2]
    
    # Crop mask to bbox
    mask = masks[index][0:1, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Dilate and erode (clean up mask)
    mask = torch.clamp(F.conv2d(mask, kernel_tensor, padding=(1, 1)), 0, 1)
    mask = torch.clamp(F.conv2d(mask, kernel_tensor * -1, padding=(1, 1)), 0, 1)
    
    # Apply mask to image crop
    cl_img = mask * img_tensor[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Shape: (3, h_bbox, w_bbox)
    
    # Center crop to 128x128
    cl_imgs[index] = pad_im_transform(cl_img)
    # Shape: (3, 128, 128)

# Result: cl_imgs shape (12, 3, 128, 128) - all cherries cropped and normalized

# ============================================
# 7. CLASSIFICATION
# ============================================
with torch.no_grad():
    classifications = my_detector.classifier(cl_imgs)
    # Input: (12, 3, 128, 128)
    # Output: (12, 2) [[logit_clean, logit_pit], ...]

# Convert to probabilities
probs = F.softmax(classifications, dim=1)
# Shape: (12, 2) [[0.95, 0.05], [0.12, 0.88], ...]

# Get best prediction
conf, classes = torch.max(probs, 1)
# conf: (12,) [0.95, 0.88, ...]
# classes: (12,) [0, 1, ...] (0=clean, 1=pit)

# ============================================
# 8. LABEL ASSIGNMENT
# ============================================
# Create boolean masks
pit_mask = probs[:, 1].ge(0.75)     # pit_prob >= 75%
maybe_mask = probs[:, 1].ge(0.5)   # pit_prob >= 50%
clean_mask = probs[:, 0].ge(0.5)     # clean_prob >= 50%

# Set final labels
prediction['labels'] = torch.where(maybe_mask, 5, prediction['labels'])
prediction['labels'] = torch.where(pit_mask, 2, prediction['labels'])
prediction['labels'] = torch.where(clean_mask, 1, prediction['labels'])

# Edge detection (conveyor sides)
prediction['labels'] = torch.where(prediction['boxes'][:, 0] < 170, 3, prediction['labels'])
prediction['labels'] = torch.where(prediction['boxes'][:, 2] > 2244, 3, prediction['labels'])

# Result: labels (12,) [1, 2, 1, 1, 3, 5, 1, ...]
# 1=clean, 2=pit, 3=side, 5=maybe

# ============================================
# 9. VISUALIZATION
# ============================================
# Create label masks
clean_mask = prediction['labels'].eq(1)
pit_mask = prediction['labels'].eq(2)
side_mask = prediction['labels'].eq(3)
maybe_mask = prediction['labels'].eq(5)

# Prepare image
img_for_labeling = (img_tensor * 255).type(torch.uint8)

# Draw boxes
img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_for_labeling,
    prediction['boxes'][clean_mask],
    colors='limegreen',
    width=2
)
img_labeled_tensor = torchvision.utils.draw_bounding_boxes(
    img_labeled_tensor,
    prediction['boxes'][pit_mask],
    colors='red',
    width=2
)
# ... continue for side, maybe

# Convert to CV2
topil = T.ToPILImage()
img_labeled = topil(img_labeled_tensor)
img_labeled = np.array(img_labeled)
img_labeled = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)

# ============================================
# 10. AGGREGATION
# ============================================
filtered = {
    'boxes': prediction['boxes'].cpu().numpy(),              # (12, 4)
    'confidences': probs.cpu().numpy(),                     # (12, 2)
    'labels': prediction['labels'].cpu().numpy()            # (12,)
}

print(f'cherries found {len(filtered)}')  # 12

# ============================================
# 11. REAL-WORLD CONVERSION
# ============================================
dets = my_detector.real_world(filtered)

# Convert pixel boxes to meters
# Rotation: 180 degrees (flip Y)
# Scale: 2710 pixels/meter
# Result: 'real_boxes' in [x1, y1, x2, y2] meters

# ============================================
# 12. ROS2 MESSAGES
# ============================================
cherries = []
for i in range(len(dets['labels'])):
    xyxy = dets['real_boxes'][i]
    cls = int(dets['labels'][i])
    
    cherry = Cherry()
    cherry.x = (xyxy[0] + xyxy[2]) / 2.0  # Center X in meters
    cherry.y = (xyxy[1] + xyxy[3]) / 2.0  # Center Y in meters
    cherry.type = (cls).to_bytes(1, 'big')  # Label as bytes
    
    cherries.append(cherry)

# Return to ROS2 service
return cherries, img_labeled
```

## Summary Table

| Stage | Input | Output | Shape | Time (GPU) |
|--------|--------|---------|--------|-------------|
| Preprocessing | CV2 (H,W,3) | Tensor (3,500,2463) | ~10ms |
| Segmentation | Tensor (1,3,500,2463) | Dict with 12 detections | ~100ms |
| Batching | 12 masks | Batch (12,3,128,128) | ~20ms |
| Classification | Batch (12,3,128,128) | Probs (12,2) | ~15ms |
| Label Assignment | Probs (12,2) | Labels (12,) | <1ms |
| Visualization | Tensor (3,500,2463) | CV2 image | ~5ms |
| **Total** | **Camera image** | **12 cherry detections** | **~150ms** |

## Key File References

| Step | File | Lines |
|------|------|-------|
| Initialization | ai_detector.py | 123-156 |
| Preprocessing | ai_detector.py | 409-420 |
| Segmentation | ai_detector.py | 427-437 |
| Classification Prep | ai_detector.py | 278-350 |
| Batching | ai_detector.py | 311-331 |
| Classification | ai_detector.py | 352-361 |
| Label Assignment | ai_detector.py | 364-382 |
| Visualization | ai_detector.py | 464-488 |
| Aggregation | ai_detector.py | 503-507 |
| Real-World | detector.py | 124-139 |
| ROS2 Messages | detector.py | 159-171 |

## Tutorial Complete

All 9 sections of the PyTorch tutorial are now available:

1. `PYTORCH_01_OVERVIEW.md` - Pipeline overview
2. `PYTORCH_02_MODELS.md` - Model definition and loading
3. `PYTORCH_03_GPU_TENSORS.md` - Device and tensor operations
4. `PYTORCH_04_PREPROCESSING.md` - Image preprocessing
5. `PYTORCH_05_INFERENCE.md` - Inference patterns
6. `PYTORCH_06_FUNCTIONAL.md` - Functional operations
7. `PYTORCH_07_TRAINING.md` - Training code examples
8. `PYTORCH_08_POSTPROCESSING.md` - Label assignment and visualization
9. `PYTORCH_09_COMPLETE_PIPELINE.md` - End-to-end walkthrough

This concludes the PyTorch tutorial for the cherry processing machine.

## Next Section

**[Back to Index](INDEX.md)**
