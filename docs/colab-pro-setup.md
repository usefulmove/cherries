# Colab Pro Setup for Cherry Classification Training

## Phase 1: Account & Access (5 min)

1. **Subscribe to Colab Pro** ($10/month via google.com/account)
2. **Ensure access to shared Google Drives**
3. **Verify you can mount the shared drives in Colab**

---

## Phase 2: Data & Code Setup (30 min)

**Data Location Question:** Which shared drive contains:
- Training images (`cherry_clean/` and `cherry_pit/` folders)?
- The original 4,000-image dataset?

**Recommended folder structure:**
```
/content/drive/Shareddrives/YOUR_SHARED_DRIVE/
├── data/
│   ├── train/
│   │   ├── cherry_clean/
│   │   └── cherry_pit/
│   └── val/
│       ├── cherry_clean/
│       └── cherry_pit/
├── cherry_classification/  # Cloned from GitHub
│   ├── models/
│   └── training script
└── checkpoints/            # For saving model weights
```

---

## Phase 3: Colab Workflow

**Quick start notebook:**
```python
# Mount drives
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Clone repo
!git clone https://github.com/weshavener/cherry_classification.git

# Change to project directory
%cd cherry_classification

# Install dependencies (if any)
!pip install -r requirements.txt

# Run training
!python train.py --data_dir /content/drive/Shareddrives/YOUR_DRIVE/data
```

---

## Phase 4: Checkpointing Strategy

**Critical:** Add periodic checkpointing to your training script:
```python
# Save every epoch
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'/content/drive/Shareddrives/YOUR_DRIVE/checkpoints/epoch_{epoch}.pt')
```
