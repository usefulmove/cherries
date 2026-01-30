#!/usr/bin/env python3
"""Quick smoke test to verify code structure without requiring full PyTorch install."""

import sys
from pathlib import Path
import yaml

print("=" * 60)
print("SMOKE TEST - Training Infrastructure")
print("=" * 60)

# Test 1: Config loading
print("\n[1/5] Testing config loading...")
try:
    with open("configs/resnet50_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert config["experiment"]["name"] == "resnet50_baseline"
    assert config["data"]["batch_size"] == 32
    assert config["training"]["epochs"] == 30
    assert config["data"]["normalize"] == True  # CRITICAL
    print("✓ Config loads correctly")
    print(f"  - Experiment: {config['experiment']['name']}")
    print(f"  - Batch size: {config['data']['batch_size']}")
    print(f"  - Epochs: {config['training']['epochs']}")
    print(f"  - ImageNet normalization: {config['data']['normalize']}")
except Exception as e:
    print(f"✗ Config test failed: {e}")
    sys.exit(1)

# Test 2: Python syntax
print("\n[2/5] Testing Python syntax...")
import py_compile

files_to_check = [
    "src/__init__.py",
    "src/data.py",
    "src/model.py",
    "src/metrics.py",
    "scripts/train.py",
    "scripts/plot_metrics.py",
]
for filepath in files_to_check:
    try:
        py_compile.compile(filepath, doraise=True)
    except Exception as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        sys.exit(1)
print("✓ All Python files have valid syntax")

# Test 3: Dataset structure
print("\n[3/5] Testing dataset structure...")
data_root = Path("../../cherry_classification/data")
if not data_root.exists():
    print(f"✗ Dataset not found at: {data_root}")
    print(
        "  Clone with: git clone https://github.com/weshavener/cherry_classification.git"
    )
    sys.exit(1)

train_clean = data_root / "train" / "cherry_clean"
train_pit = data_root / "train" / "cherry_pit"
val_clean = data_root / "val" / "cherry_clean"
val_pit = data_root / "val" / "cherry_pit"

for path in [train_clean, train_pit, val_clean, val_pit]:
    if not path.exists():
        print(f"✗ Missing directory: {path}")
        sys.exit(1)
    count = len(list(path.glob("*")))
    print(f"  - {path.name}: {count} images")

print("✓ Dataset structure is correct")

# Test 4: Output directory creation
print("\n[4/5] Testing output directory creation...")
test_output = Path("test_output")
test_output.mkdir(exist_ok=True)
assert test_output.exists()
print(f"✓ Output directory creation works: {test_output}")

# Test 5: Module structure
print("\n[5/5] Testing module imports (will fail on torch, that's okay)...")
sys.path.insert(0, str(Path.cwd()))
try:
    # This will fail due to missing torch, but we can catch it
    from src import data, model, metrics

    print("✓ Modules import successfully (PyTorch is installed)")
except ModuleNotFoundError as e:
    if "torch" in str(e) or "sklearn" in str(e) or "matplotlib" in str(e):
        print(f"⚠ Module structure is correct, but dependencies missing: {e}")
        print(
            "  This is expected - install with: pip install torch torchvision pyyaml scikit-learn matplotlib"
        )
    else:
        print(f"✗ Unexpected import error: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("SMOKE TEST RESULTS")
print("=" * 60)
print("✓ Config parsing: PASS")
print("✓ Python syntax: PASS")
print("✓ Dataset structure: PASS")
print("✓ Directory operations: PASS")
print("⚠ Dependencies: Not installed (expected)")
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("The training infrastructure is structurally sound!")
print("\nNext steps:")
print(
    "1. Install dependencies: pip install torch torchvision pyyaml scikit-learn matplotlib"
)
print(
    "2. Run training: python scripts/train.py --config configs/resnet50_baseline.yaml \\"
)
print(
    "                                        --data-root ../cherry_classification/data \\"
)
print("                                        --output-dir ./test_run")
print("\nOr use Google Colab for GPU training (see notebooks/colab_runner.ipynb)")
print("=" * 60)
