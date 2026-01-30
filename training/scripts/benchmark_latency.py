import torch
import time
import argparse
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_classifier


def benchmark(model, device, batch_size=1, num_runs=100, input_size=128):
    """Measure inference latency."""

    # Generate dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # Warmup
    print(f"Warming up ({10} runs)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            # Sync CUDA if needed for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return latencies


def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Latency")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model .pt file"
    )
    parser.add_argument(
        "--architecture", type=str, default="resnet50", help="Model architecture"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu/cuda)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\nBenchmarking: {args.architecture}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model Path: {args.model_path}")

    # Load Model
    try:
        model = create_classifier(
            architecture=args.architecture,
            num_classes=2,
            pretrained=False,
            device=args.device,
        )
        state_dict = torch.load(args.model_path, map_location=device)
        # Handle state dict keys if they are wrapped or different
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run Benchmark
    latencies = benchmark(model, device, batch_size=args.batch_size)

    # Report
    latencies = np.array(latencies)
    print("\n" + "=" * 40)
    print(f"RESULTS ({args.num_runs if 'num_runs' in args else 100} runs)")
    print("-" * 40)
    print(f"Mean Latency:   {np.mean(latencies):.2f} ms")
    print(f"Median Latency: {np.median(latencies):.2f} ms")
    print(f"Min Latency:    {np.min(latencies):.2f} ms")
    print(f"Max Latency:    {np.max(latencies):.2f} ms")
    print(f"P99 Latency:    {np.percentile(latencies, 99):.2f} ms")
    print(f"Throughput:     {1000 / np.mean(latencies) * args.batch_size:.2f} fps")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
