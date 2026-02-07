"""
Phase 2 Experiment Runner for Google Colab

This module contains ready-to-run cells for executing Phase 2 experiments.
Copy these cells into your Colab notebook and run in order.

Experiments:
- EXP-001: Threshold Optimization (CPU only)
- EXP-002A: ConvNeXt-Tiny Baseline
- EXP-002B: ConvNeXt-Tiny with Label Smoothing
- EXP-003A: EfficientNet-B2 Baseline
- EXP-003B: EfficientNet-B2 with Label Smoothing

Usage:
1. Copy cell contents into Colab notebook cells
2. Set SKIP_COMPLETED to control which experiments run
3. Run all cells
4. Download results from Google Drive
"""

# ============================================================
# CELL 1: Configuration - Set which experiments to run
# ============================================================

EXPERIMENT_CONFIG = {
    # Skip flags - set to True to skip already completed experiments
    "SKIP_EXP_001": False,  # Threshold optimization
    "SKIP_EXP_002A": False,  # ConvNeXt-Tiny baseline
    "SKIP_EXP_002B": False,  # ConvNeXt-Tiny label smoothing
    "SKIP_EXP_003A": False,  # EfficientNet-B2 baseline
    "SKIP_EXP_003B": False,  # EfficientNet-B2 label smoothing
    
    # Paths (adjust for your setup)
    "DATA_PATH": "/content/cherry_classification/data",
    "OUTPUT_BASE": "/content/drive/MyDrive/cherry_experiments",
    "BASELINE_MODEL": "/content/drive/MyDrive/cherry_experiments/resnet50_augmented_unnormalized/model_best.pt",
}

# Check which experiments will run
print("=" * 60)
print("EXPERIMENT CONFIGURATION")
print("=" * 60)
for exp, skip in EXPERIMENT_CONFIG.items():
    if exp.startswith("SKIP_"):
        status = "SKIP" if skip else "RUN"
        print(f"{exp}: {status}")
print("=" * 60)

# ============================================================
# CELL 2: GPU Check (for training experiments)
# ============================================================

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU available! Training will be very slow.")
    print("Consider upgrading to Colab Pro or using CPU for EXP-001 only.")

# ============================================================
# CELL 3: Setup - Mount Drive and Clone Repo
# ============================================================

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Create output directory
os.makedirs(EXPERIMENT_CONFIG["OUTPUT_BASE"], exist_ok=True)

# Clone repos (if not already done)
!if [ ! -d "/content/traina" ]; then git clone https://github.com/dedmonds/traina.git /content/traina; fi
!if [ ! -d "/content/cherry_classification" ]; then git clone https://github.com/dedmonds/cherry_classification.git /content/cherry_classification; fi

# Add training scripts to path
import sys
sys.path.insert(0, '/content/traina/training')

print("Setup complete!")
print(f"Output directory: {EXPERIMENT_CONFIG['OUTPUT_BASE']}")

# ============================================================
# CELL 4: EXP-001 - Threshold Optimization
# ============================================================

if not EXPERIMENT_CONFIG["SKIP_EXP_001"]:
    print("\n" + "=" * 60)
    print("RUNNING EXP-001: Threshold Optimization")
    print("=" * 60)
    
    # Check if baseline model exists
    baseline_model = EXPERIMENT_CONFIG["BASELINE_MODEL"]
    if not os.path.exists(baseline_model):
        print(f"ERROR: Baseline model not found: {baseline_model}")
        print("Please ensure the 94.05% ResNet50 model is uploaded to Drive")
    else:
        output_dir = f"{EXPERIMENT_CONFIG['OUTPUT_BASE']}/threshold_optimization"
        
        !python /content/traina/training/scripts/optimize_thresholds.py \
            --model-path {baseline_model} \
            --data-root {EXPERIMENT_CONFIG["DATA_PATH"]} \
            --architecture resnet50 \
            --output-dir {output_dir} \
            --min-recall 0.99 \
            --device cpu
        
        print(f"\nResults saved to: {output_dir}")
        print("Download threshold_results.json and threshold_analysis.png")
else:
    print("EXP-001: SKIPPED (set SKIP_EXP_001=False to run)")

# ============================================================
# CELL 5: EXP-002A - ConvNeXt-Tiny Baseline (Seed 42)
# ============================================================

if not EXPERIMENT_CONFIG["SKIP_EXP_002A"]:
    print("\n" + "=" * 60)
    print("RUNNING EXP-002A: ConvNeXt-Tiny Baseline")
    print("=" * 60)
    
    # Set random seed for reproducibility
    import random
    import numpy as np
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config_path = "/content/traina/training/configs/experiments/convnext_tiny_baseline_seed42.yaml"
    
    # Update output dir in config to use Drive
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['checkpointing']['output_dir'] = f"{EXPERIMENT_CONFIG['OUTPUT_BASE']}/convnext_tiny_baseline_seed42"
    config['data']['root'] = EXPERIMENT_CONFIG["DATA_PATH"]
    
    # Save modified config
    temp_config = "/tmp/convnext_tiny_baseline_seed42.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    !python /content/traina/training/scripts/train.py \
        --config {temp_config} \
        --data-root {EXPERIMENT_CONFIG["DATA_PATH"]}
    
    print(f"\nResults saved to: {config['checkpointing']['output_dir']}")
else:
    print("EXP-002A: SKIPPED (set SKIP_EXP_002A=False to run)")

# ============================================================
# CELL 6: EXP-002B - ConvNeXt-Tiny with Label Smoothing
# ============================================================

if not EXPERIMENT_CONFIG["SKIP_EXP_002B"]:
    print("\n" + "=" * 60)
    print("RUNNING EXP-002B: ConvNeXt-Tiny with Label Smoothing")
    print("=" * 60)
    
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config_path = "/content/traina/training/configs/experiments/convnext_tiny_label_smooth_seed42.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['checkpointing']['output_dir'] = f"{EXPERIMENT_CONFIG['OUTPUT_BASE']}/convnext_tiny_label_smooth_seed42"
    config['data']['root'] = EXPERIMENT_CONFIG["DATA_PATH"]
    
    temp_config = "/tmp/convnext_tiny_label_smooth_seed42.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    !python /content/traina/training/scripts/train.py \
        --config {temp_config} \
        --data-root {EXPERIMENT_CONFIG["DATA_PATH"]}
    
    print(f"\nResults saved to: {config['checkpointing']['output_dir']}")
else:
    print("EXP-002B: SKIPPED (set SKIP_EXP_002B=False to run)")

# ============================================================
# CELL 7: EXP-003A - EfficientNet-B2 Baseline
# ============================================================

if not EXPERIMENT_CONFIG["SKIP_EXP_003A"]:
    print("\n" + "=" * 60)
    print("RUNNING EXP-003A: EfficientNet-B2 Baseline")
    print("=" * 60)
    
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config_path = "/content/traina/training/configs/experiments/efficientnet_b2_baseline_seed42.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['checkpointing']['output_dir'] = f"{EXPERIMENT_CONFIG['OUTPUT_BASE']}/efficientnet_b2_baseline_seed42"
    config['data']['root'] = EXPERIMENT_CONFIG["DATA_PATH"]
    
    temp_config = "/tmp/efficientnet_b2_baseline_seed42.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    !python /content/traina/training/scripts/train.py \
        --config {temp_config} \
        --data-root {EXPERIMENT_CONFIG["DATA_PATH"]}
    
    print(f"\nResults saved to: {config['checkpointing']['output_dir']}")
else:
    print("EXP-003A: SKIPPED (set SKIP_EXP_003A=False to run)")

# ============================================================
# CELL 8: EXP-003B - EfficientNet-B2 with Label Smoothing
# ============================================================

if not EXPERIMENT_CONFIG["SKIP_EXP_003B"]:
    print("\n" + "=" * 60)
    print("RUNNING EXP-003B: EfficientNet-B2 with Label Smoothing")
    print("=" * 60)
    
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config_path = "/content/traina/training/configs/experiments/efficientnet_b2_label_smooth_seed42.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['checkpointing']['output_dir'] = f"{EXPERIMENT_CONFIG['OUTPUT_BASE']}/efficientnet_b2_label_smooth_seed42"
    config['data']['root'] = EXPERIMENT_CONFIG["DATA_PATH"]
    
    temp_config = "/tmp/efficientnet_b2_label_smooth_seed42.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    !python /content/traina/training/scripts/train.py \
        --config {temp_config} \
        --data-root {EXPERIMENT_CONFIG["DATA_PATH"]}
    
    print(f"\nResults saved to: {config['checkpointing']['output_dir']}")
else:
    print("EXP-003B: SKIPPED (set SKIP_EXP_003B=False to run)")

# ============================================================
# CELL 9: Post-Experiment Analysis
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

# List all completed experiments
import json
from pathlib import Path

base_path = Path(EXPERIMENT_CONFIG["OUTPUT_BASE"])

print("\nCompleted Experiments:")
print("-" * 60)

for exp_dir in sorted(base_path.glob("*")):
    if exp_dir.is_dir():
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            # Read last line for final metrics
            with open(metrics_file) as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    acc = last_entry.get('accuracy', 'N/A')
                    epoch = last_entry.get('epoch', 'N/A')
                    print(f"{exp_dir.name:40s} | Epoch {epoch:2s} | Acc: {acc}")

print("-" * 60)
print(f"\nAll results saved to: {EXPERIMENT_CONFIG['OUTPUT_BASE']}")
print("Download metrics.json and model_best.pt files for analysis")

# ============================================================
# CELL 10: Compare All Models (if multiple experiments completed)
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Find all model_best.pt files
model_files = list(Path(EXPERIMENT_CONFIG["OUTPUT_BASE"]).rglob("model_best.pt"))

if len(model_files) > 1:
    print(f"\nFound {len(model_files)} trained models for comparison")
    print("Run comparison with: scripts/compare_models.py")
    
    # Create model list for comparison
    model_list = " ".join([f"--models {str(m.parent.name)}={str(m)}" for m in model_files])
    
    print(f"\nCommand to run comparison locally:")
    print(f"python scripts/compare_models.py {model_list} --data-root {EXPERIMENT_CONFIG['DATA_PATH']}")
else:
    print("Only 1 model found. Need at least 2 for comparison.")

# ============================================================
# END OF COLAB EXPERIMENT RUNNER
# ============================================================
