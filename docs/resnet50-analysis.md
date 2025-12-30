## Analysis of the ResNet50 Classifier Architecture

Based on my review of `ai_detector.py`, here's the current classifier design:

### Current Architecture

| Component | Implementation |
|-----------|----------------|
| **Base Model** | ResNet50 (pre-trained on ImageNet) |
| **Final Layer** | `nn.Linear(2048, 2)` - Binary classification |
| **Input Size** | 128×128 cropped cherry images |
| **Preprocessing** | Mask applied to crop, then `CenterCrop(128)` |
| **Normalization** | **Missing** - no ImageNet normalization applied |
| **Thresholds** | pit≥0.75, maybe≥0.5, clean≥0.5 |

---

## Potential Improvements to Increase Accuracy

### 1. **Missing Input Normalization (High Priority)**

**Problem:** The inference code at `ai_detector.py:329-331` does NOT apply ImageNet normalization before classification:

```python
cl_img = ((masks[index])[0:1,bbox[1]:bbox[3],bbox[0]:bbox[2]]) * (img_tensor)[0:3,bbox[1]:bbox[3],bbox[0]:bbox[2]]
cl_imgs[index] = pad_im_transform(cl_img)  # Only CenterCrop, no normalization!
```

ResNet50 was pre-trained with normalized inputs (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). If training used normalization but inference doesn't (or vice versa), this creates a distribution mismatch.

**Recommendation:** Ensure consistent normalization between training and inference:
```python
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
cl_imgs[index] = normalize(pad_im_transform(cl_img))
```

---

### 2. **ResNet50 May Be Overkill (Medium Priority)**

**Problem:** ResNet50 has 25.6M parameters and was designed for 1000-class ImageNet classification. For binary cherry classification on 128×128 images, this is likely overparameterized and prone to overfitting with limited data.

**Recommendations:**
- **ResNet18** (11.7M params) - Often performs similarly on small datasets with less overfitting
- **EfficientNet-B0** (5.3M params) - Better accuracy/parameter tradeoff
- **MobileNetV3** (5.4M params) - Optimized for smaller inputs

---

### 3. **Consider Finetuning More Layers (Medium Priority)**

**Current approach** (per documentation): Only the final `fc` layer is trained:
```python
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

**Problem:** Cherry pit detection is a specialized visual task. The frozen ImageNet features may not capture pit-specific textures optimally.

**Recommendations:**
- Unfreeze the last 1-2 residual blocks (`layer4`, possibly `layer3`)
- Use differential learning rates: lower LR for earlier layers, higher for later layers
- Example:
  ```python
  optimizer = optim.Adam([
      {'params': model.layer4.parameters(), 'lr': 1e-4},
      {'params': model.fc.parameters(), 'lr': 1e-3}
  ])
  ```

---

### 4. **Data Augmentation During Training (Medium Priority)**

The documentation shows minimal augmentation:
```python
T.RandomHorizontalFlip(0.5)
T.ColorJitter(brightness=0.2, contrast=0.2)
```

**Recommendations for cherry classification:**
- **Random rotation** (full 360°) - cherries can appear at any angle
- **Random vertical flip** - cherries have no "up"
- **Gaussian blur** - simulate focus variation
- **Random erasing/cutout** - improve robustness to occlusions
- **Affine transforms** - handle perspective variations

```python
train_transform = T.Compose([
    T.RandomRotation(180),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.RandomErasing(p=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

### 5. **Class Imbalance Handling (Medium Priority)**

If clean vs. pit samples are imbalanced (common in production data where most cherries are clean):

**Recommendations:**
- Use weighted `CrossEntropyLoss`:
  ```python
  weights = torch.tensor([1.0, 3.0])  # Higher weight for minority (pit) class
  criterion = nn.CrossEntropyLoss(weight=weights.to(device))
  ```
- Oversample minority class in DataLoader
- Use Focal Loss for hard example mining

---

### 6. **Input Resolution (Low-Medium Priority)**

**Current:** 128×128 pixels

**Problem:** Pits may require finer detail to detect reliably. ResNet50 was designed for 224×224.

**Recommendation:** Test with 224×224 crops:
- Better feature extraction from pretrained weights
- More detail for pit detection
- Trade-off: Slightly slower inference

---

### 7. **Replace with Attention Mechanisms (Low Priority - Experimental)**

**Option A:** Add a Squeeze-and-Excitation (SE) block after the backbone:
```python
# After ResNet features, before FC
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)
```

**Option B:** Use Vision Transformer (ViT-Tiny) - may capture global pit patterns better than CNNs.

---

### 8. **Adjust Decision Thresholds (Quick Win)**

**Current thresholds:**
- pit ≥ 0.75 → label 2 (pit)
- maybe ≥ 0.5 → label 5 (uncertain)
- clean ≥ 0.5 → label 1 (clean)

**Problem:** These thresholds may not be optimal. The "maybe" category creates ambiguity.

**Recommendations:**
- Analyze the probability distributions on validation data
- Use precision-recall curves to find optimal threshold
- Consider eliminating the "maybe" category or adjusting the gap between 0.5 and 0.75

---

## Summary of Recommendations (Prioritized)

| Priority | Change | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| **High** | Fix normalization mismatch | +2-5% | Low |
| **High** | Expand data augmentation | +1-3% | Low |
| **Medium** | Finetune more layers | +1-3% | Medium |
| **Medium** | Handle class imbalance | +1-2% | Low |
| **Medium** | Try smaller model (ResNet18) | Reduce overfitting | Medium |
| **Low** | Increase input resolution to 224 | +0.5-1% | Low |
| **Low** | Tune decision thresholds | +0.5-1% | Low |
