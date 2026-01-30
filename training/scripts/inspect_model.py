import torch
import sys
import os
import time
import argparse
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_classifier


def inspect_model(model_path, device_name="cpu"):
    print(f"\n--- Inspecting Model: {model_path} ---")
    device = torch.device(device_name)

    # Load Weights to check format
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return

    print(f"Loading weights...")
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Handle checkpoint format
    if "model_state_dict" in state_dict:
        print("Format: Training Checkpoint (contains 'model_state_dict')")
        weights = state_dict["model_state_dict"]
    else:
        print("Format: Raw State Dict (Weights Only)")
        weights = state_dict

    # Create Skeleton Model for Inference Test
    # We default to ResNet50 as it's the main production model,
    # but strictly speaking we should match architecture.
    # For inspection of weights, we can just look at the tensor dict first.

    print(f"\nModel Statistics:")
    total_params = sum(p.numel() for p in weights.values() if torch.is_tensor(p))
    print(f"  Total Parameters in weights: {total_params:,}")

    # Check for Denormals
    print("\nScanning for denormal values (potential latency killer)...")
    denormal_count = 0
    tensors_with_denormals = 0

    for name, param in weights.items():
        if not torch.is_tensor(param) or not param.is_floating_point():
            continue

        # Check for denormals: non-zero values smaller than 1e-35 (approx)
        # float32 normal range min is ~1.18e-38, but denormals trigger slowdowns below that.
        # We use a safe threshold.
        mask = (param.abs() > 0) & (param.abs() < 1e-32)
        if mask.any():
            count = mask.sum().item()
            if count > 0:
                print(f"  WARNING: {name} has {count} denormal values!")
                denormal_count += count
                tensors_with_denormals += 1

    if denormal_count == 0:
        print("  PASS: No denormal values detected.")
    else:
        print(
            f"  FAIL: Found {denormal_count} denormal values in {tensors_with_denormals} tensors."
        )
        print("  Recommendation: Run 'fix_denormals.py' on this model.")

    # Try to load into model for single inference test
    print("\nRunning single inference test...")
    try:
        # Assumption: ResNet50. If this fails, user might need to specify arch.
        model = create_classifier(
            architecture="resnet50", num_classes=2, pretrained=False, device=device_name
        )
        model.load_state_dict(
            weights, strict=False
        )  # Strict false in case of minor mismatch
        model.eval()

        # Check memory format of first layer
        first_param = next(model.parameters())
        is_channels_last = first_param.is_contiguous(memory_format=torch.channels_last)
        print(
            f"  Memory Format: {'Channels Last' if is_channels_last else 'Contiguous (Standard)'}"
        )

        # Run inference
        dummy = torch.rand(1, 3, 128, 128) * 255  # Simulate raw input
        if device_name == "cuda":
            dummy = dummy.cuda()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        print(f"  Single Inference Time: {(end - start) * 1000:.2f} ms")

    except Exception as e:
        print(f"  Could not run inference test (Architecture mismatch?): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model weights for health and latency issues."
    )
    parser.add_argument("model_path", help="Path to the .pt model file")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")

    args = parser.parse_args()

    inspect_model(args.model_path, args.device)


if __name__ == "__main__":
    main()
