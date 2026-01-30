# Cherry Pit Classifier Training

This directory contains the training infrastructure for the ResNet50 binary classifier that detects cherry pits.

## Directory Structure

```
training/
├── src/                    # Core training modules
│   ├── data.py            # Data loading and preprocessing
│   ├── model.py           # Model creation and checkpoint management
│   └── metrics.py         # Metrics calculation and logging
├── scripts/                # Training scripts
│   ├── train.py           # Main training script
│   └── plot_metrics.py    # Visualization utility
├── configs/                # Configuration files
│   └── resnet50_baseline.yaml
├── notebooks/              # Jupyter notebooks
│   └── colab_runner.ipynb # Google Colab training notebook
└── README.md              # This file
```

## Quick Start

### Option 1: Local Training (CPU)

1. **Clone the dataset repository:**
   ```bash
   cd /path/to/repos/
   git clone https://github.com/weshavener/cherry_classification.git
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision pyyaml scikit-learn matplotlib
   ```

3. **Run training:**
   ```bash
   cd /path/to/traina
   python training/scripts/train.py \
       --config training/configs/resnet50_baseline.yaml \
       --data-root ../cherry_classification/data \
       --output-dir ./training_outputs/baseline_run1
   ```

### Option 2: Google Colab Pro (GPU, Recommended)

1. **Open the Colab notebook:**
   - Upload `training/notebooks/colab_runner.ipynb` to Google Colab
   - Or open directly: [Open in Colab](https://colab.research.google.com/)

2. **Run all cells sequentially:**
   - The notebook handles repo cloning, dependency installation, and training
   - Models are automatically saved to your Google Drive
   - Expected runtime: 1-2 hours for 30 epochs

## Dataset Information

**Source:** https://github.com/weshavener/cherry_classification

**Structure:**
```
data/
├── train/
│   ├── cherry_clean/    (1,978 images - label 0)
│   └── cherry_pit/      (1,698 images - label 1)
└── val/
    ├── cherry_clean/
    └── cherry_pit/
```

**Class Balance:** 53.8% clean / 46.2% pit (well-balanced, no weighted loss needed)

## Configuration

Training parameters are defined in YAML config files. The baseline config (`configs/resnet50_baseline.yaml`) contains:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 128 | Image size (matches inference) |
| `batch_size` | 32 | Samples per batch |
| `epochs` | 30 | Number of training epochs |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `pretrained` | true | Use ImageNet pretrained weights |
| `normalize` | true | **ImageNet normalization (CRITICAL)** |

### Creating Custom Configs

Copy the baseline config and modify parameters:

```bash
cp training/configs/resnet50_baseline.yaml training/configs/my_experiment.yaml
# Edit my_experiment.yaml
python training/scripts/train.py --config training/configs/my_experiment.yaml ...
```

## Training Outputs

After training, the output directory contains:

```
outputs/resnet50_baseline/
├── config.yaml                # Copy of config for reproducibility
├── metrics.json               # Training metrics (JSON Lines format)
├── model_best.pt              # Best model (highest val accuracy)
├── model_final.pt             # Final model (last epoch)
├── checkpoint_epoch_5.pt      # Periodic checkpoints
├── checkpoint_epoch_10.pt
├── ...
└── training_curves.png        # Loss/accuracy plots
```

### File Formats

**Model Files (`.pt`):**
- `model_best.pt` and `model_final.pt`: State dict only (for inference)
- `checkpoint_epoch_*.pt`: Full training state (for resuming)

**Metrics Log (`metrics.json`):**
- JSON Lines format (one JSON object per line)
- Each line contains metrics for one epoch/phase
- Use `plot_metrics.py` to visualize

## Monitoring Training

### During Training

Watch real-time console output:
```
Epoch [1] Batch [10/55] Loss: 0.543
Epoch [1] Batch [20/55] Loss: 0.421
...
```

Or monitor the log file:
```bash
tail -f outputs/baseline/metrics.json
```

### After Training

Generate training curves:
```bash
python training/scripts/plot_metrics.py outputs/baseline/metrics.json
```

This creates `outputs/baseline/training_curves.png` with loss and accuracy plots.

## Resuming from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
python training/scripts/train.py \
    --config training/configs/resnet50_baseline.yaml \
    --data-root ../cherry_classification/data \
    --output-dir ./outputs/baseline \
    --resume ./outputs/baseline/checkpoint_epoch_25.pt
```

## Advanced Usage

### Testing Individual Modules

Each module has a `__main__` block for standalone testing:

```bash
# Test data loading
python training/src/data.py ../cherry_classification/data

# Test model creation
python training/src/model.py

# Test metrics calculation
python training/src/metrics.py
```

### Custom Training Loop

Import modules directly in your own scripts:

```python
from training.src.data import get_dataloaders
from training.src.model import create_resnet50_classifier
from training.src.metrics import calculate_metrics

# Your custom training logic here
```

## Troubleshooting

### Common Issues

**1. "Data root not found" error:**
```
FileNotFoundError: Data root not found: ../cherry_classification/data
```
**Solution:** Clone the dataset repo:
```bash
git clone https://github.com/weshavener/cherry_classification.git
```

**2. Out of memory (CUDA OOM):**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `batch_size` in the config file (try 16 instead of 32).

**3. Slow training on CPU:**
Training on CPU takes ~10x longer than GPU. Consider using Google Colab Pro for GPU access.

**4. Import errors:**
```
ModuleNotFoundError: No module named 'yaml'
```
**Solution:** Install missing dependencies:
```bash
pip install pyyaml scikit-learn matplotlib
```

### Validation Accuracy Too Low (< 80%)

Possible causes:
1. **Normalization bug:** Verify ImageNet normalization is enabled in config (`normalize: true`)
2. **Data loading issue:** Check that images are loading correctly
3. **Learning rate too high/low:** Try adjusting `learning_rate` in config
4. **Model not learning:** Check console output for loss decreasing

### Training Too Slow

**On Colab:**
- Verify GPU is enabled: `Runtime → Change runtime type → Hardware accelerator → GPU`
- Check GPU usage: Run `!nvidia-smi` in a Colab cell

**Locally:**
- Use smaller `batch_size` if GPU memory is limited
- Reduce `num_workers` if CPU is bottleneck
- Consider using mixed precision training (set `mixed_precision: true` in config)

## Dependencies

**Required:**
- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- PyYAML
- scikit-learn
- numpy

**Optional:**
- matplotlib (for plotting)
- Google Colab (for GPU training)

Install all at once:
```bash
pip install torch torchvision pyyaml scikit-learn matplotlib numpy
```

## Next Steps

After training completes:

1. **Evaluate the model:**
   - Check validation accuracy (target: > 90%)
   - Review confusion matrix for class-specific performance
   - View training curves for signs of overfitting

2. **Deploy the model:**
   - Copy `model_best.pt` to the ROS2 package
   - Update `cherry_detection` to load the new weights
   - Test on real hardware

3. **Iterate if needed:**
   - Enable data augmentation: `augmentation: true` in config
   - Try different learning rates or optimizers
   - Increase training epochs if underfitting

## References

- **Dataset:** https://github.com/weshavener/cherry_classification
- **Model Architecture:** ResNet50 (He et al., 2015)
- **Transfer Learning:** ImageNet pretrained weights
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Support

For issues or questions:
- Check `docs/core/ARCHITECTURE.md` for system overview
- Review `docs/stories/STORY-001-Training-Infrastructure.md` for implementation details
- See `docs/reference/resnet50-analysis.md` for model improvement suggestions
