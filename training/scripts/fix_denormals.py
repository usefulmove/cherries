import torch
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_classifier


def fix_denormals(model_path, output_path, threshold=1e-32):
    print(f"Loading {model_path}...")
    device = torch.device("cpu")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # Handle checkpoint format
    if "model_state_dict" in state_dict:
        weights = state_dict["model_state_dict"]
        is_checkpoint = True
    else:
        weights = state_dict
        is_checkpoint = False

    print(f"Scanning {len(weights)} parameters/buffers...")
    count = 0
    fixed_params = 0

    for key, tensor in weights.items():
        if not torch.is_tensor(tensor):
            continue

        if not tensor.is_floating_point():
            continue

        # Check for denormals
        mask = (tensor.abs() > 0) & (tensor.abs() < threshold)
        if mask.any():
            count += mask.sum().item()
            fixed_params += 1
            # Fix in place
            tensor[mask] = 0.0

    print(f"Fixed {count} denormal values across {fixed_params} tensors.")

    if is_checkpoint:
        state_dict["model_state_dict"] = weights
        torch.save(state_dict, output_path)
    else:
        torch.save(weights, output_path)

    print(f"Saved fixed model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input model path")
    parser.add_argument("output", help="Output model path")
    args = parser.parse_args()

    fix_denormals(args.input, args.output)
