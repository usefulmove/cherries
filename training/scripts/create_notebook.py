#!/usr/bin/env python3
"""Create a valid Phase 2 Colab notebook with proper JSON structure."""

import json


def create_notebook():
    cells = []

    # Cell 0: Title
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 2: SOTA Optimization Experiments\n",
                "\n",
                "This notebook executes the updated Phase 2 experimental roadmap incorporating external research feedback:\n",
                "\n",
                "**New State-of-the-Art Experiments:**\n",
                "1. **EXP-001**: Threshold Optimization (CPU, 4-6 hours)\n",
                "2. **EXP-002A**: ConvNeXt V2-Tiny Baseline (GPU, ~12 hours) - *FCMAE pre-training*\n",
                "3. **EXP-002B**: ConvNeXt V2-Tiny + Label Smoothing (GPU, ~12 hours)\n",
                "4. **EXP-003A**: EfficientNet-B2 Baseline (GPU, ~10 hours) - *Speed-focused*\n",
                "5. **EXP-003B**: EfficientNet-B2 + Label Smoothing (GPU, ~10 hours)\n",
                "6. **EXP-006A**: DINOv2 ViT-S/14 Linear Probe (GPU, ~4 hours) - *Foundation model*\n",
                "\n",
                "**Key Improvements:**\n",
                "- **Enhanced augmentations**: Motion blur + stronger color jitter for conveyor realism\n",
                "- **ConvNeXt V2**: Upgraded from V1 to V2 with FCMAE pre-training (better for defects)\n",
                "- **DINOv2**: Foundation model approach with frozen backbone + linear probe\n",
                "\n",
                "**Current Baseline:**\n",
                "- ResNet50: 94.05% accuracy, 16ms latency, 25.6M params\n",
                "\n",
                "**Target:**\n",
                "- Accuracy: ≥94.5% (stretch: 95%)\n",
                "- Pit Recall: ≥99.0% (food safety)\n",
                "- Latency: <30ms on CPU\n",
                "\n",
                "**Prerequisites:**\n",
                "- Google Colab Pro (GPU required for training experiments)\n",
                "- Baseline model uploaded to Google Drive\n",
                "- Data in Drive at: `cherry_classification/data/`",
            ],
        }
    )

    # Cell 1: Config Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 1: Experiment Configuration\n",
                "\n",
                "Set skip flags to control which experiments run. Use `SMOKE_TEST=True` for quick validation (1 epoch, 3 batches).",
            ],
        }
    )

    # Cell 2: Configuration Code
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# === EXPERIMENT CONFIGURATION ===\n",
                "# Set these flags to control execution\n",
                "\n",
                "SMOKE_TEST = False  # Set True for quick validation (1 epoch, 3 batches)\n",
                "\n",
                "# Skip flags - set True to skip already completed experiments\n",
                "SKIP_EXP_001 = False    # Threshold optimization (CPU only)\n",
                "SKIP_EXP_002A = False   # ConvNeXt V2-Tiny baseline (NEW - Phase 2)\n",
                "SKIP_EXP_002B = False   # ConvNeXt V2-Tiny + label smoothing (NEW)\n",
                "SKIP_EXP_003A = False   # EfficientNet-B2 baseline\n",
                "SKIP_EXP_003B = False   # EfficientNet-B2 + label smoothing\n",
                "SKIP_EXP_006A = False   # DINOv2 ViT-S/14 linear probe (NEW - Phase 2)\n",
                "\n",
                "# Random seed for reproducibility\n",
                "RANDOM_SEED = 42\n",
                "\n",
                "# Paths (adjust for your Drive structure)\n",
                'DRIVE_MOUNT_PATH = "/content/drive"\n',
                'DRIVE_BASE_PATH = "/content/drive/MyDrive/cherry_experiments"\n',
                'DATA_PATH = "/content/cherry_classification/data"\n',
                'BASELINE_MODEL_PATH = "/content/drive/MyDrive/cherry_experiments/resnet50_augmented_unnormalized/model_best_fixed.pt"\n',
                "\n",
                "# Print configuration\n",
                'print("=" * 60)\n',
                'print("EXPERIMENT CONFIGURATION - Phase 2 SOTA")\n',
                'print("=" * 60)\n',
                'print(f"SMOKE_TEST: {SMOKE_TEST}")\n',
                'print(f"RANDOM_SEED: {RANDOM_SEED}")\n',
                'print("\\nPhase 2 NEW Experiments:")\n',
                'print(f"  EXP-002A (ConvNeXt V2 baseline): {SKIP_EXP_002A}")\n',
                'print(f"  EXP-002B (ConvNeXt V2 + LS): {SKIP_EXP_002B}")\n',
                'print(f"  EXP-006A (DINOv2 linear probe): {SKIP_EXP_006A}")\n',
                'print("\\nOther Experiments:")\n',
                'print(f"  EXP-001 (Threshold opt): {SKIP_EXP_001}")\n',
                'print(f"  EXP-003A (EfficientNet B2): {SKIP_EXP_003A}")\n',
                'print(f"  EXP-003B (EfficientNet B2 + LS): {SKIP_EXP_003B}")\n',
                'print("=" * 60)',
            ],
        }
    )

    # Cell 3: Dependencies Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 2: Environment Setup & Dependencies\n",
                "\n",
                "Install required packages including `timm` for ConvNeXt V2 support.",
            ],
        }
    )

    # Cell 4: Dependencies Code
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                'print("Installing dependencies...")\n',
                "!pip install -q pyyaml scikit-learn matplotlib tqdm\n",
                "\n",
                "# NEW: Install timm for ConvNeXt V2\n",
                'print("Installing timm for ConvNeXt V2...")\n',
                "!pip install -q timm\n",
                "\n",
                "# Verify installations\n",
                "import importlib\n",
                "\n",
                'print("\\n" + "=" * 60)\n',
                'print("DEPENDENCY CHECK")\n',
                'print("=" * 60)\n',
                "\n",
                "# Check timm\n",
                "try:\n",
                "    import timm\n",
                '    print(f"✓ timm installed: {timm.__version__}")\n',
                "except ImportError:\n",
                '    print("✗ timm not available - ConvNeXt V2 will fail")\n',
                "\n",
                "# Check torch\n",
                "import torch\n",
                'print(f"✓ PyTorch: {torch.__version__}")\n',
                'print("=" * 60)',
            ],
        }
    )

    # Cell 5: GPU Check Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Cell 3: GPU Check & Drive Mount"],
        }
    )

    # Cell 6: GPU Check Code
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Mount Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount(DRIVE_MOUNT_PATH, force_remount=True)\n",
                "\n",
                "# Check GPU availability\n",
                'print("\\n" + "=" * 60)\n',
                'print("GPU CHECK")\n',
                'print("=" * 60)\n',
                'print(f"CUDA available: {torch.cuda.is_available()}")\n',
                "\n",
                "if not torch.cuda.is_available():\n",
                "    # Check if we're running training experiments\n",
                "    needs_gpu = not (SKIP_EXP_002A and SKIP_EXP_002B and SKIP_EXP_003A and SKIP_EXP_003B and SKIP_EXP_006A)\n",
                "    \n",
                "    if needs_gpu:\n",
                "        raise RuntimeError(\n",
                '            "\\n" + "!" * 60 + "\\n" +\n',
                '            "GPU REQUIRED FOR TRAINING EXPERIMENTS!\\n" +\n',
                '            "Go to: Runtime -> Change runtime type -> GPU\\n" +\n',
                '            "Then re-run this cell.\\n" +\n',
                '            "!" * 60\n',
                "        )\n",
                "    else:\n",
                '        print("WARNING: No GPU available, but only running EXP-001 (CPU). Continuing...")\n',
                "else:\n",
                '    print(f"CUDA version: {torch.version.cuda}")\n',
                '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
                '    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")\n',
                "\n",
                'print("=" * 60)',
            ],
        }
    )

    # Cell 7: Clone Repos Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Cell 4: Clone Repositories"],
        }
    )

    # Cell 8: Clone Repos Code
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone training repository\n",
                '!if [ ! -d "/content/traina" ]; then \\\n',
                "    git clone https://github.com/dedmonds/traina.git /content/traina; \\\n",
                "fi\n",
                "\n",
                "# Clone dataset repository (shallow clone)\n",
                '!if [ ! -d "/content/cherry_classification" ]; then \\\n',
                "    git clone --depth 1 https://github.com/weshavener/cherry_classification.git /content/cherry_classification; \\\n",
                "fi\n",
                "\n",
                "# Add training scripts to path\n",
                "import sys\n",
                "sys.path.insert(0, '/content/traina/training')\n",
                "\n",
                "# Create output directories\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "output_dirs = [\n",
                '    f"{DRIVE_BASE_PATH}/threshold_optimization",\n',
                '    f"{DRIVE_BASE_PATH}/convnextv2_tiny_baseline_seed42",\n',
                '    f"{DRIVE_BASE_PATH}/convnextv2_tiny_label_smooth_seed42",\n',
                '    f"{DRIVE_BASE_PATH}/efficientnet_b2_baseline_seed42",\n',
                '    f"{DRIVE_BASE_PATH}/efficientnet_b2_label_smooth_seed42",\n',
                '    f"{DRIVE_BASE_PATH}/dinov2_vits14_linear_probe_seed42",\n',
                "]\n",
                "\n",
                "for dir_path in output_dirs:\n",
                "    Path(dir_path).mkdir(parents=True, exist_ok=True)\n",
                "    \n",
                'print("\\n" + "=" * 60)\n',
                'print("SETUP COMPLETE")\n',
                'print("=" * 60)\n',
                'print(f"Training code: /content/traina")\n',
                'print(f"Data: {DATA_PATH}")\n',
                'print(f"Output base: {DRIVE_BASE_PATH}")\n',
                "\n",
                "# Verify data exists\n",
                "if not os.path.exists(DATA_PATH):\n",
                '    print(f"\\nWARNING: Data not found at {DATA_PATH}")\n',
                "else:\n",
                '    print(f"\\nData verified: {DATA_PATH}")\n',
                "    !ls -lh {DATA_PATH}\n",
                'print("=" * 60)',
            ],
        }
    )

    # Cell 9: EXP-001 Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cell 5: EXP-001 - Threshold Optimization\n",
                "\n",
                "**Type:** Analysis (no training)\n",
                "**Duration:** 4-6 hours\n",
                "**Requires:** Baseline model (94.05% ResNet50)\n",
                "\n",
                "Find optimal decision boundaries for 3-class classification (clean/pit/maybe).",
            ],
        }
    )

    # Cell 10: EXP-001 Code
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if SKIP_EXP_001:\n",
                '    print("EXP-001: SKIPPED (set SKIP_EXP_001=False to run)")\n',
                "else:\n",
                '    print("\\n" + "=" * 60)\n',
                '    print("RUNNING EXP-001: Threshold Optimization")\n',
                '    print("=" * 60)\n',
                "    \n",
                "    # Verify baseline model exists\n",
                "    if not os.path.exists(BASELINE_MODEL_PATH):\n",
                '        print(f"ERROR: Baseline model not found: {BASELINE_MODEL_PATH}")\n',
                '        print("Please upload the 94.05% ResNet50 model to Google Drive")\n',
                '        print("Expected location: MyDrive/cherry_experiments/resnet50_augmented_unnormalized/model_best_fixed.pt")\n',
                "    else:\n",
                '        output_dir = f"{DRIVE_BASE_PATH}/threshold_optimization"\n',
                "        \n",
                "        # Run threshold optimization\n",
                "        !python /content/traina/training/scripts/optimize_thresholds.py \\\n",
                "            --model-path {BASELINE_MODEL_PATH} \\\n",
                "            --data-root {DATA_PATH} \\\n",
                "            --architecture resnet50 \\\n",
                "            --output-dir {output_dir} \\\n",
                "            --min-recall 0.99 \\\n",
                "            --device cpu\n",
                "        \n",
                '        print("\\n" + "=" * 60)\n',
                '        print(f"Results saved to: {output_dir}")\n',
                '        print("\\nKey files to download:")\n',
                '        print("  - threshold_results.json")\n',
                '        print("  - optimal_thresholds.yaml")\n',
                '        print("  - threshold_analysis.png")\n',
                '        print("  - probability_distributions.png")\n',
                '        print("=" * 60)',
            ],
        }
    )

    # Continue with remaining cells...
    # I'll create a helper function to add the rest

    return cells


if __name__ == "__main__":
    cells = create_notebook()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open("training/notebooks/colab_phase2_experiments.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)

    print("✓ Notebook structure created")
    print(f"  Total cells: {len(cells)}")
