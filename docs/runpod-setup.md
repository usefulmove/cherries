## ResNet50 Training Setup for RunPod (A100 PCIe)

### 1. Connect to RunPod and Clone Repo
```bash
git clone <your-repo-url>
cd traina/cherry_system
```

### 2. Create Python Environment
```bash
python3 -m venv cherry_env
source cherry_env/bin/activate

# Install PyTorch with CUDA for A100
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install opencv-python numpy Pillow matplotlib pandas scikit-learn

# Install ROS2 (optional - only needed for full system)
sudo apt update
sudo apt install ros-humble-desktop -y
```

### 3. Prepare Training Data
Create this structure:
```
data/
├── train/
│   ├── clean/      # Cherry images without pits
│   └── pit/        # Cherry images with pits
├── val/
│   ├── clean/
│   └── pit/
└── test/
    ├── clean/
    └── pit/
```
Or use a labels CSV with columns: `filename,label` (0=clean, 1=pit)

**Note:** If you don't have the original annotated data, you'll need to:
- Export images from your production system, or
- Manually curate a dataset from cherry images

### 4. ResNet50 Training Script
```python
# train_resnet50.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

# Config
DATA_DIR = Path("data")
BATCH_SIZE = 64  # A100 can handle larger batches
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class CherryDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.df.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

train_loader = DataLoader(CherryDataset("train.csv", DATA_DIR/"train", train_transform), 
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CherryDataset("val.csv", DATA_DIR/"val", val_transform), 
                        batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation...
    print(f"Epoch {epoch+1}/{EPOCHS}")

# Save
torch.save(model.state_dict(), "cherry_classification.pt")
```

### 5. Run Training
```bash
source cherry_env/bin/activate
python train_resnet50.py
```

### 6. Deploy Trained Model
```bash
# Copy new weights to both locations
cp cherry_classification.pt cherry_system/cherry_detection/resource/
cp cherry_classification.pt cherry_system/control_node/resource/
```
