# PyTorch Tutorial: Cherry Processing Machine - Section 7: Training

**Note:** The cherry processing codebase only contains inference code. This section provides training examples based on the model architectures used.

## Mask R-CNN Training

### Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json
from PIL import Image
import numpy as np
import os

class CherrySegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.annotations = json.load(open(annotation_file))
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotation (masks, boxes, labels)
        ann = self.annotations[self.images[idx]]
        masks = np.array(ann['masks'])  # [N, H, W]
        boxes = np.array(ann['boxes'])    # [N, 4] in [x1, y1, x2, y2]
        labels = np.array(ann['labels'])  # [N]

        # Convert to tensors
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # Target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
```

### Data Transforms

```python
import torchvision.transforms as T

# Training transforms
def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    return T.Compose(transforms)
```

### Training Loop

```python
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 1. Load model (same as inference)
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,  # Load COCO weights for transfer learning
        num_classes=num_classes,
        min_size=800,
        max_size=2464,
    )
    return model

# 2. Setup dataset and dataloader
dataset = CherrySegmentationDataset(
    image_dir='data/train/images',
    annotation_file='data/train/annotations.json',
    transforms=get_transform(train=True)
)

dataset_test = CherrySegmentationDataset(
    image_dir='data/val/images',
    annotation_file='data/val/annotations.json',
    transforms=get_transform(train=False)
)

data_loader = DataLoader(
    dataset,
    batch_size=2,  # Mask R-CNN uses small batches
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))  # Custom collate for variable-sized tensors
)

data_loader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))
)

# 3. Setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 4. Create model
num_classes = 2  # background + cherry
model = get_instance_segmentation_model(num_classes)
model.to(device)

# 5. Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# 6. Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# 7. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Mask R-CNN returns 4 losses
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}')

# 8. Save weights
torch.save(model.state_dict(), 'cherry_segmentation.pt')
```

**Mask R-CNN losses:**
- `loss_classifier`: Classification loss (background vs cherry)
- `loss_box_reg`: Bounding box regression loss
- `loss_objectness`: Region proposal loss
- `loss_mask`: Segmentation mask loss

## ResNet50 Training

### Dataset Class

```python
class CherryClassificationDataset(Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.samples = []
        
        # Load labels: {filename: label} where 0=clean, 1=pit
        self.labels = json.load(open(label_file))
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        label = self.labels[self.images[idx]]  # 0 or 1

        if self.transforms:
            img = self.transforms(img)

        return img, label
```

### Fine-Tuning Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

# 1. Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# 2. Modify final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: clean, pit

# 3. Freeze backbone (optional)
for param in model.parameters():
    param.requires_grad = False

# Only train the final layer
for param in model.fc.parameters():
    param.requires_grad = True

# 4. Setup transforms
train_transform = T.Compose([
    T.RandomResizedCrop(128),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Setup dataset
train_dataset = CherryClassificationDataset(
    'data/train/crops',
    'data/train/labels.json',
    transforms=train_transform
)

val_dataset = CherryClassificationDataset(
    'data/val/crops',
    'data/val/labels.json',
    transforms=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. Setup device and model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 7. Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 8. Training loop
num_epochs = 20
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total

    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'cherry_classification.pt')
```

## Training vs. Inference Summary

| Aspect | Inference | Training |
|--------|-----------|-----------|
| Mode | `model.eval()` | `model.train()` |
| Gradients | `torch.no_grad()` | Compute gradients |
| Dataset | Single images | DataLoader |
| Batching | Optional | Required |
| Loss | None | Required |
| Optimizer | None | Required |
| Augmentation | None | Random |

## Next Section

**`PYTORCH_08_POSTPROCESSING.md`** - Label assignment, bounding boxes, result aggregation
