# PyTorch Tutorial: Cherry Processing Machine - Section 3: GPU & Tensors

## Device Selection

**Reference:** `ai_detector.py:127`

```python
# Automatically detect GPU
self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

**Key functions:**
- `torch.cuda.is_available()`: Returns `True` if CUDA GPU is detected
- `torch.device('cuda')` or `torch.device('cpu')`: Create device object
- `.to(device)`: Move models/tensors to specified device

## Moving Models to GPU

**Reference:** `ai_detector.py:141, 148, 156`

```python
# After model creation
self.model.to(self.device)           # Mask R-CNN to GPU
self.classifier = resnet50().to(self.device)  # ResNet50 to GPU during creation
self.classifier.cuda()               # Alternative: explicit CUDA placement
```

**Best practice:** Always move model to same device as input tensors.

## Tensor Creation

### From NumPy

**Reference:** `ai_detector.py:307`

```python
import numpy as np

# 5x5 convolution kernel
kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]], dtype=np.float32)

# Convert to tensor and add batch/channel dimensions
kernel_tensor = torch.tensor(
    np.expand_dims(np.expand_dims(kernel, 0), 0),  # Shape: (1, 1, 5, 5)
    device='cuda'
)
```

**Key functions:**
- `torch.tensor(array)`: Convert NumPy array to tensor
- `np.expand_dims()`: Add dimensions before conversion
- `device='cuda'`: Create tensor directly on GPU

### Zero Allocation

**Reference:** `ai_detector.py:311`

```python
# Pre-allocate tensor for batch of cherry crops
# Shape: (num_cherries, channels, height, width)
cl_imgs = torch.zeros(size_masks[0], 3, 128, 128, device='cuda')
```

**Key functions:**
- `torch.zeros(*shape, device)`: Create zero-filled tensor
- `size_masks[0]`: Number of detected cherries
- Shape convention: `(batch, channels, height, width)`

## Tensor Indexing & Slicing

### Bounding Box Slicing

**Reference:** `ai_detector.py:317, 325, 329`

```python
# Get bounding box for each detection
bbox = boxes[index].type(torch.int)  # Convert float bbox to int indices

# Slice mask using bbox coordinates
# mask shape: [1, H, W], bbox: [x1, y1, x2, y2]
mask = (masks[index])[0:1, bbox[1]:bbox[3], bbox[0]:bbox[2]]

# Slice image using same bbox
cl_img = (img_tensor)[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
```

**Slicing syntax:** `tensor[channel_start:channel_end, row_start:row_end, col_start:col_end]`

### Channel Selection

```python
# Get first channel only
mask = masks[index][0:1, ...]  # Shape: [1, H, W]

# Get RGB channels
img = img_tensor[0:3, ...]     # Shape: [3, H, W]
```

## Tensor Operations on GPU

### Element-wise Operations

**Reference:** `ai_detector.py:326-327`

```python
# Dilate: Conv2D with all-ones kernel
mask = torch.clamp(
    torch.nn.functional.conv2d(mask, kernel_tensor, padding=(1, 1)),
    0, 1
)

# Erode: Conv2D with negative kernel
mask = torch.clamp(
    torch.nn.functional.conv2d(mask, kernel_tensor * -1, padding=(1, 1)),
    0, 1
)
```

### Masking

**Reference:** `ai_detector.py:329`

```python
# Multiply image by mask (element-wise)
cl_img = masks[index][0:1, bbox[1]:bbox[3], bbox[0]:bbox[2]] * img_tensor[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
```

### Type Conversion

**Reference:** `ai_detector.py:471`

```python
# Convert float [0, 1] to uint8 [0, 255]
img_for_labeling = (img_tensor * 255).type(torch.uint8)
```

## CPU-GPU Transfer

### Tensor to CPU

**Reference:** `ai_detector.py:494-499`

```python
# Move tensors to CPU for NumPy conversion/visualization
boxes = prediction['boxes'].cpu().numpy()
labels = prediction['labels'].cpu().numpy()
confidence_probs = prediction['confidence_probs'].cpu().numpy()
```

**Key functions:**
- `.cpu()`: Move tensor to CPU (creates copy)
- `.numpy()`: Convert PyTorch tensor to NumPy array
- Note: `.numpy()` only works on CPU tensors

### Device Consistency

```python
# All tensors in an operation must be on same device
# GPU tensors cannot operate with CPU tensors

# ✅ Correct: Both on CUDA
result = gpu_tensor1 * gpu_tensor2

# ❌ Error: Mixed devices
result = gpu_tensor * cpu_tensor
```

## Tensor Shape Reference

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `img_tensor` | `[3, H, W]` | RGB image (CHW format) |
| `masks` | `[N, 1, H, W]` | N segmentation masks |
| `boxes` | `[N, 4]` | N bounding boxes |
| `cl_imgs` | `[N, 3, 128, 128]` | N cherry crops for classification |
| `kernel_tensor` | `[1, 1, 5, 5]` | Convolution kernel |

**N = number of detected cherries**

## Next Section

**[04. Preprocessing](PYTORCH_04_PREPROCESSING.md)** - Image preprocessing and transforms
