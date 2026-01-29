# PyTorch Tutorial: Cherry Processing Machine - Section 4: Preprocessing

## CV2/PIL to Tensor Conversion

**Reference:** `ai_detector.py:92-98`

```python
import torchvision.transforms as T
import cv2

def pil_to_tensor_gpu(pil_image, device):
    # Convert PIL image to tensor
    img_to_tensor = T.ToTensor()
    img_tensor = img_to_tensor(pil_image)
    
    # Move to GPU
    gpu_tensor = img_tensor.to(device)
    return gpu_tensor
```

**In detect() method:**

```python
# Convert CV2 (BGR) to PIL (RGB)
img_pil = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

# Convert to tensor on GPU
img_tensor = pil_to_tensor_gpu(img_pil, self.device)
```

**Key points:**
- CV2 uses BGR color order, PIL uses RGB
- `T.ToTensor()`: Converts `[H, W, C] uint8 [0, 255]` → `[C, H, W]` float32 [0.0, 1.0]
- Shape transformation: Height-Width-Channels → Channels-Height-Width

## Transforms

### PILToTensor

**Reference:** `ai_detector.py:144`

```python
# Convert tensor back to PIL (for visualization)
self.tesnor_to_img = T.PILToTensor()
```

### CenterCrop

**Reference:** `ai_detector.py:297`

```python
# Crop cherry regions to 128x128
pad_im_transform = T.transforms.CenterCrop(128)

# Apply to each cherry crop
cl_imgs[index] = pad_im_transform(cl_img)
```

### ToPILImage

**Reference:** `ai_detector.py:467, 484`

```python
# Convert tensor to PIL for drawing
topil = T.ToPILImage()

# Prepare image for bounding boxes
img_for_labeling = (img_tensor * 255).type(torch.uint8)
img_labeled = topil(img_labeled_tensor)
```

## Common Transforms

```python
import torchvision.transforms as T

# Resize
resize = T.Resize((256, 256))

# Normalize (for training)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Random augmentations (training)
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])
```

## Image Cropping with Bounding Boxes

**Reference:** `ai_detector.py:317, 329`

```python
# Get bounding box as integers
bbox = boxes[index].type(torch.int)  # [x1, y1, x2, y2]

# Crop image using bbox coordinates
# bbox[1]:bbox[3] = y1 to y2 (rows)
# bbox[0]:bbox[2] = x1 to x2 (columns)
cl_img = img_tensor[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
```

## Batch Processing for Classification

**Reference:** `ai_detector.py:311-331`

```python
# Pre-allocate batch tensor
cl_imgs = torch.zeros(size_masks[0], 3, 128, 128, device='cuda')

# Process each cherry
for index, mask in enumerate(masks):
    bbox = boxes[index].type(torch.int)
    
    # Apply mask to image
    mask = masks[index][0:1, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cl_img = mask * img_tensor[0:3, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Center crop to 128x128
    cl_imgs[index] = pad_im_transform(cl_img)
```

## Training Data Augmentation Example

```python
import torchvision.transforms as T

# Training transforms with augmentation
train_transform = T.Compose([
    T.ToPILImage(),               # Convert tensor to PIL
    T.RandomHorizontalFlip(p=0.5), # 50% chance flip
    T.RandomVerticalFlip(p=0.5),   # 50% chance flip
    T.RandomRotation(15),         # ±15 degree rotation
    T.ColorJitter(               # Color variations
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    T.ToTensor(),                # Convert back to tensor
    T.Normalize(                 # Standardize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/test transforms (no augmentation)
val_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Preprocessing Pipeline

```
Raw CV2 Image (BGR)
       │
       ▼ cv2.cvtColor
PIL Image (RGB)
       │
       ▼ T.ToTensor()
Tensor [C, H, W], float [0, 1]
       │
       ▼ .to(device)
GPU Tensor
       │
       ▼ Mask R-CNN
Segmentation (boxes, masks)
       │
       ▼ Crop + Transform
Batch [N, 3, 128, 128]
       │
       ▼ ResNet50
Classification
```

## Next Section

**`PYTORCH_05_INFERENCE.md`** - Inference patterns and no_grad context
