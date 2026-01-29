# PyTorch Tutorial: Cherry Processing Machine - Section 2: Models

## Model Definition

### Mask R-CNN: Instance Segmentation

**Reference:** `ai_detector.py:30-75`

```python
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False,          # Don't load COCO weights
        num_classes=2,             # Background + cherry matter
        
        # Transform parameters
        min_size=800,              # Minimum image dimension
        max_size=2464,             # Maximum image dimension (matches camera)
        
        # RPN (Region Proposal Network) parameters
        rpn_pre_nms_top_n_train=20000,
        rpn_pre_nms_top_n_test=10000,
        rpn_post_nms_top_n_train=20000,
        rpn_post_nms_top_n_test=10000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        
        # Box detection parameters
        box_score_thresh=0.5,      # Minimum detection confidence
        box_nms_thresh=0.5,        # Non-maximum suppression
        box_detections_per_img=1000,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=5120,
        box_positive_fraction=0.25,
    )
    return model
```

**Key concepts:**
- `maskrcnn_resnet50_fpn`: Pre-built Mask R-CNN with ResNet50 backbone + Feature Pyramid Network
- `pretrained=False`: Loading custom trained weights, not COCO
- `num_classes=2`: Binary segmentation (background vs cherry)
- RPN parameters control region proposal quality
- Box parameters filter final detections

### ResNet50: Binary Classification

**Reference:** `ai_detector.py:148-156`

```python
# Load pre-trained ResNet50
self.classifier = resnet50().to(self.device)

# Get input features of final layer
num_ftrs = self.classifier.fc.in_features

# Replace final layer for 2-class output (clean vs pit)
self.classifier.fc = torch.nn.Linear(num_ftrs, 2)

# Load trained weights
self.classifier.load_state_dict(weights2)

# Set to evaluation mode
self.classifier.eval()
```

**Key concepts:**
- `resnet50()`: 50-layer residual network (ImageNet pre-trained)
- `fc`: Final fully-connected layer, replaced for custom classification
- `in_features`: Number of inputs to FC layer (2048 for ResNet50)
- `torch.nn.Linear`: Create new output layer with 2 outputs

## Model Loading

### Initialization in Detector Class

**Reference:** `ai_detector.py:123-142`

```python
class ai_detector_class:
    def __init__(self, weights, weights2):
        # Determine device (GPU if available)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Create segmentation model
        self.model = get_instance_segmentation_model(2)
        
        # Load segmentation weights
        self.model.load_state_dict(weights)
        self.model.eval()                    # Set to inference mode
        self.model.to(self.device)           # Move to GPU
        
        # Create and load classifier
        self.classifier = resnet50().to(self.device)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 2)
        self.classifier.load_state_dict(weights2)
        self.classifier.eval()              # Set to inference mode
        self.classifier.cuda()              # Explicit GPU placement
```

### Loading Weights from File

**Reference:** `ai_detector.py:522-525`

```python
from ament_index_python.packages import get_package_share_directory

# Get package directory
package_share_directory = get_package_share_directory('cherry_detection')

# Load segmentation weights
weight_path = os.path.join(package_share_directory, 'segmentation_20.pt')
weights = torch.load(weight_path)

# Load classification weights
weight_path2 = os.path.join(package_share_directory, 'cherry_classification.pt')
weights2 = torch.load(weight_path)

# Initialize detector with weights
my_detector = ai_detector_class(weights, weights2)
```

**Key functions:**
- `torch.load()`: Loads saved model weights from `.pt` file
- `load_state_dict()`: Applies weights to model architecture
- `.eval()`: Sets model to inference mode (disables dropout/batch norm updates)
- `.to(device)`: Moves model to CPU or GPU

## Eval Mode vs Train Mode

```python
model.eval()  # Inference: Faster, no gradients, batch norm uses running stats
model.train()  # Training: Computes gradients, enables dropout, updates batch norm
```

**Always use `.eval()` before inference** to ensure consistent behavior.

## Model Output Formats

### Mask R-CNN Output

```python
with torch.no_grad():
    predictions = self.model(img_tensor.unsqueeze(0))

prediction = predictions[0]  # Single image output
```

Output dictionary contains:
- `boxes`: `[N, 4]` - Bounding boxes `[x1, y1, x2, y2]`
- `labels`: `[N]` - Predicted class IDs
- `scores`: `[N]` - Confidence scores
- `masks`: `[N, 1, H, W]` - Pixel-level masks

**Reference:** `ai_detector.py:429-437`

### ResNet50 Output

```python
with torch.no_grad():
    classifications = self.classifier(cl_imgs)

probs = torch.nn.functional.softmax(classifications, dim=1)
conf, classes = torch.max(probs, 1)
```

- `classifications`: `[N, 2]` - Raw logits (pre-softmax)
- `probs`: `[N, 2]` - Softmax probabilities `[p_clean, p_pit]`
- `conf`: `[N]` - Max probability per sample
- `classes`: `[N]` - Predicted class (0 or 1)

**Reference:** `ai_detector.py:355-368`

## Next Section

**`PYTORCH_03_GPU_TENSORS.md`** - Device management, tensor creation, and indexing
