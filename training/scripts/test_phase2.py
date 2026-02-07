#!/usr/bin/env python3
"""
Phase 2 Experiment Validation Script
Run this before deploying to Colab to verify all dependencies, configs, and model architectures.
"""

import sys
import os
import json
import yaml
import warnings
import traceback
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Suppress expected warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
try:
    from src.model import create_classifier, DinoV2Classifier
    from src.data import get_transforms, get_dataloaders

    print("✓ Layer 2: Modules imported successfully")
except ImportError as e:
    print(f"✗ Layer 2 FAILED: Import error - {e}")
    sys.exit(1)


def test_notebook_syntax():
    """Layer 1: Validate notebook JSON structure"""
    print("\n[Layer 1] Validating Notebook Syntax...")
    nb_path = Path("training/notebooks/colab_phase2_experiments.ipynb")
    try:
        with open(nb_path, "r") as f:
            json.load(f)
        print("✓ Notebook JSON parses cleanly")
    except json.JSONDecodeError as e:
        print(f"✗ Notebook JSON error: {e}")
        sys.exit(1)


def test_configs():
    """Layer 5: Validate Config Schemas"""
    print("\n[Layer 5] Validating Configurations...")
    configs = [
        "training/configs/experiments/convnextv2_tiny_baseline_seed42.yaml",
        "training/configs/experiments/dinov2_vits14_linear_probe_seed42.yaml",
        "training/configs/experiments/efficientnet_b2_baseline_seed42.yaml",
    ]

    required_keys = ["model", "training", "data", "checkpointing"]

    for cfg_path in configs:
        if not os.path.exists(cfg_path):
            print(f"⚠ Warning: Config not found: {cfg_path}")
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Schema check
        for key in required_keys:
            if key not in cfg:
                print(f"✗ Config {cfg_path} missing key: {key}")
                sys.exit(1)

        # Optimizer check
        opt = cfg["training"].get("optimizer", "adam")
        if opt not in ["adam", "sgd", "adamw"]:
            print(f"✗ Unsupported optimizer '{opt}' in {cfg_path}")
            sys.exit(1)

    print(f"✓ {len(configs)} configs validated")


def test_models():
    """Layer 3: Validate Model Instantiation & Forward Pass"""
    print("\n[Layer 3] Validating Model Architectures...")

    # 1. ResNet50 (Baseline)
    try:
        model = create_classifier(
            "resnet50", num_classes=2, pretrained=False, device="cpu"
        )
        out = model(torch.randn(1, 3, 128, 128))
        assert out.shape == (1, 2)
        print("✓ ResNet50: Ready")
    except Exception as e:
        print(f"✗ ResNet50 Failed: {e}")
        sys.exit(1)

    # 2. ConvNeXt V2
    try:
        model = create_classifier(
            "convnextv2_tiny", num_classes=2, pretrained=False, device="cpu"
        )
        out = model(torch.randn(1, 3, 128, 128))
        assert out.shape == (1, 2)
        print("✓ ConvNeXt V2: Ready")
    except Exception as e:
        print(f"✗ ConvNeXt V2 Failed: {e}")
        sys.exit(1)

    # 3. DINOv2 (Resolution Check)
    try:
        model = create_classifier(
            "dinov2_vits14", num_classes=2, pretrained=False, device="cpu"
        )

        # Test 1: Native 224 (Should PASS)
        out = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 2)

        # Test 2: Custom 126 (Should PASS)
        out = model(torch.randn(1, 3, 126, 126))
        assert out.shape == (1, 2)

        print("✓ DINOv2: Ready (Verified at 224x224 and 126x126)")

        # Test 3: Invalid 128 (Should FAIL)
        try:
            model(torch.randn(1, 3, 128, 128))
            print("✗ DINOv2 Error: Should have failed at 128x128 but didn't")
            sys.exit(1)
        except (AssertionError, RuntimeError):
            print("✓ DINOv2: Correctly rejected invalid resolution 128x128")

    except Exception as e:
        print(f"✗ DINOv2 Failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def test_transforms():
    """Layer 4: Validate Data Augmentation Pipeline"""
    print("\n[Layer 4] Validating Enhanced Transforms...")

    # Create dummy image
    img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))

    # Test 128 pipeline
    tf_128, _ = get_transforms(input_size=128, augmentation=True)
    out = tf_128(img)
    assert out.shape == (3, 128, 128)

    # Test 126 pipeline (DINO)
    tf_126, _ = get_transforms(input_size=126, augmentation=True)
    out = tf_126(img)
    assert out.shape == (3, 126, 126)

    print("✓ Transform pipeline functional (Motion blur + ColorJitter)")


def test_adamw_support():
    """Layer 5: Validate Optimizer Integration"""
    print("\n[Layer 5] Validating AdamW Support...")
    model = nn.Linear(10, 2)
    try:
        import torch.optim as optim

        opt = optim.AdamW(model.parameters(), lr=1e-3)
        print("✓ AdamW optimizer instantiated successfully")
    except AttributeError:
        print("✗ AdamW optimizer NOT found in torch.optim")
        sys.exit(1)


if __name__ == "__main__":
    print("=== Phase 2 Validation Suite ===\n")

    test_notebook_syntax()
    test_configs()
    test_models()
    test_transforms()
    test_adamw_support()

    print("\n=== ALL SYSTEMS GO: Ready for Colab Deployment ===")
