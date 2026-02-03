# PyTorch Tutorial: Cherry Processing Machine - Section 5: Inference

## torch.no_grad() Context

**Reference:** `ai_detector.py:430, 355`

```python
# Disable gradient computation for inference
with torch.no_grad():
    predictions = self.model(img_tensor.unsqueeze(0))
    classifications = self.classifier(cl_imgs)
```

**Why use `torch.no_grad()`:**
- Faster: No gradient tracking overhead
- Less memory: Gradients not stored
- Inference: Only forward pass needed

**Always use during inference.**

## Model Forward Pass

### Mask R-CNN Inference

**Reference:** `ai_detector.py:429-437`

```python
with torch.no_grad():
    # Add batch dimension: [C, H, W] → [1, C, H, W]
    predictions = self.model(img_tensor.unsqueeze(0))

# Get single image prediction
prediction = predictions[0]
```

**Model output:**
```python
{
    'boxes': torch.Tensor,      # [N, 4] Bounding boxes
    'labels': torch.Tensor,      # [N] Class IDs
    'scores': torch.Tensor,      # [N] Detection confidence
    'masks': torch.Tensor,      # [N, 1, H, W] Segmentation masks
}
```

### ResNet50 Classification

**Reference:** `ai_detector.py:355-357`

```python
with torch.no_grad():
    classifications = self.classifier(cl_imgs)
```

**Model output:**
```python
classifications  # [N, 2] Logits (pre-softmax)
```

## Batch Inference

**Reference:** `ai_detector.py:311-331`

```python
# Pre-allocate batch tensor
cl_imgs = torch.zeros(size_masks[0], 3, 128, 128, device='cuda')

# Build batch from individual cherries
for index, mask in enumerate(masks):
    bbox = boxes[index].type(torch.int)
    cl_img = mask * img_tensor[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cl_imgs[index] = pad_im_transform(cl_img)

# Process entire batch at once (faster than individual inference)
with torch.no_grad():
    classifications = self.classifier(cl_imgs)
```

**Benefits of batching:**
- Parallel GPU utilization
- Faster than N sequential forward passes
- Better memory efficiency

## Add Batch Dimension

```python
# Single image
img_tensor.shape  # [3, 500, 2463]

# Add batch dimension
img_tensor.unsqueeze(0).shape  # [1, 3, 500, 2463]

# Alternative methods
img_tensor[None, ...]           # [1, 3, 500, 2463]
img_tensor.view(1, -1, *img_tensor.shape[1:])  # [1, 3, 500, 2463]
```

## Inference vs. Training

### Inference

```python
model.eval()

with torch.no_grad():
    output = model(input)
    # No backward pass
    # No optimizer step
```

### Training

```python
model.train()

output = model(input)
loss = criterion(output, target)
loss.backward()              # Compute gradients
optimizer.step()             # Update weights
optimizer.zero_grad()        # Clear gradients
```

## Profiling Inference

**Reference:** `ai_detector.py:424-425, 433-434`

```python
import time

prep_for_seg_start = time.time()
# ... preprocessing
prep_for_seg_end = time.time()
print(f'prep for start; {prep_for_seg_end - prep_for_seg_start}')

prediction_start = time.time()
with torch.no_grad():
    predictions = self.model(img_tensor.unsqueeze(0))
prediction_end = time.time()
print(f'predict; {prediction_end - prediction_start}')
```

**Typical timings (GPU):**
- Preprocessing: ~10-50ms
- Mask R-CNN: ~50-150ms
- ResNet50 classification: ~10-30ms per batch

## Inference Pipeline

```
Input Image
     │
     ▼ Convert
Tensor on GPU
     │
     ▼ unsqueeze(0)
[1, C, H, W]
     │
     ▼ with torch.no_grad()
Mask R-CNN Forward
     │
     ▼ Output
boxes, masks, scores
     │
     ▼ Crop + Batch
[N, 3, 128, 128]
     │
     ▼ with torch.no_grad()
ResNet50 Forward
     │
     ▼ Output
classifications
```

## Next Section

**[06. Functional Operations](PYTORCH_06_FUNCTIONAL.md)** - Functional operations (conv2d, softmax, torch.where)
