---
name: training-colab
description: Execute model training workflows on Google Colab with Google Drive integration for the Cherry Processing System.
---

# Training on Google Colab Skill

This skill enables efficient model training using Google Colab's GPU resources while maintaining code and data synchronization with the local project via Google Drive.

## When to Use

- Need GPU acceleration for model training
- Training experiments that exceed local hardware capabilities
- Long-running training jobs that benefit from cloud execution
- Testing hyperparameter variations at scale

## Prerequisites

1. Google account with Colab Pro (recommended) or Colab Free
2. Google Drive with project folder structure
3. Training data uploaded to Google Drive
4. Local project configured for Drive sync (see `docs/reference/colab-pro-setup.md`)

## Workflow

### Step 1: Prepare Training Configuration

Create or update a training config in `training/configs/`:

```yaml
# training/configs/experiment_name.yaml
model:
  name: resnet50
  num_classes: 2
  pretrained: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  
data:
  train_path: /content/drive/MyDrive/traina/data/train
  val_path: /content/drive/MyDrive/traina/data/val
  
augmentation:
  enabled: true
  normalize: false  # Set based on production requirements
```

### Step 2: Sync Code to Google Drive

Before syncing, perform a **Pre-flight Check**:
1. Run a local smoke test (e.g., `python smoke_test.py` or set `DRY_RUN=True` in your notebook).
2. Verify that 1 epoch runs on CPU with a few batches.
3. Only sync after local verification passes.

```bash
# From project root
./training/scripts/sync_to_drive.sh
```

### Step 3: Launch Colab Notebook

1. Open `training/notebooks/colab_training.ipynb` in Google Colab
2. Connect to GPU runtime (Runtime → Change runtime type → GPU)
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Execute training cells

### Step 4: Download Results

After training completes:

```bash
# Sync trained models back from Drive
./training/scripts/sync_from_drive.sh \
  --source "drive/MyDrive/traina/experiments/" \
  --dest "training/experiments/"
```

## Configuration Options

### Environment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Development** | Quick iterations, small dataset | Testing configs, debugging |
| **Training** | Full dataset, GPU acceleration | Production model training |
| **Evaluation** | Validation/test metrics only | Assessing trained models |

### GPU Optimization

```python
# Enable mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Best Practices

1. **Version Control**
   - Git tracks code and configs locally
   - Drive stores datasets and model weights
   - Never commit large `.pt` files to git

2. **Experiment Tracking**
   - Use descriptive experiment names
   - Log all hyperparameters
   - Save training curves and metrics

3. **Data Management**
   - Keep training data in Drive under versioned folders
   - Use symbolic links for large datasets
   - Validate data integrity before training

4. **Cost Optimization**
   - Colab Free: Limited GPU hours per day
   - Colab Pro: Faster GPUs, longer sessions
   - Monitor runtime to avoid losing progress

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Drive not mounting | Re-authenticate, check permissions |
| CUDA out of memory | Reduce batch_size, use gradient accumulation |
| Training interrupted | Enable checkpointing, resume from last epoch |
| Slow data loading | Pre-cache dataset to local Colab storage |

## Related Resources

- [PyTorch Training Tutorial](../../core/framework/pytorch/PYTORCH_07_TRAINING.md)
- [Colab Pro Setup Guide](../../reference/colab-pro-setup.md)
- [Benchmark Latency Skill](../benchmark-latency/SKILL.md)
- [Evaluate Model Skill](../evaluate-model/SKILL.md)
