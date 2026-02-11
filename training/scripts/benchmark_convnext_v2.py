#!/usr/bin/env python3
"""
Evaluate ConvNeXt V2 model locally and measure latency.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import time
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import timm


def load_model(model_path, device):
    """Load ConvNeXt V2 model from checkpoint."""
    model = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
    }


def measure_latency(
    model, device, num_warmup=10, num_runs=100, input_size=(1, 3, 224, 224)
):
    """Measure inference latency on CPU."""
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


def main():
    # Setup
    device = torch.device("cpu")  # Force CPU for latency measurement
    model_path = Path(
        "/home/dedmonds/repos/traina/temp-phase2-experiments/convnextv2_tiny_baseline_seed42/model_best.pt"
    )
    data_path = Path(
        "/home/dedmonds/repos/traina/cherry_system/src/cherry_detection/resource/dataset_12_16_24"
    )

    print("=" * 60)
    print("ConvNeXt V2 Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)
    print("Model loaded successfully")

    # Check if data exists
    if not data_path.exists():
        print(f"\nData path not found: {data_path}")
        print("Skipping accuracy evaluation - will measure latency only")

        # Measure latency only
        print("\nMeasuring latency (CPU, 100 runs)...")
        latency = measure_latency(model, device)
        print(f"\nLatency Results:")
        print(f"  Mean:   {latency['mean']:.2f} ms")
        print(f"  Median: {latency['median']:.2f} ms")
        print(f"  Std:    {latency['std']:.2f} ms")
        print(f"  Min:    {latency['min']:.2f} ms")
        print(f"  Max:    {latency['max']:.2f} ms")
        return

    # Prepare data (unnormalized like production)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # No normalization
        ]
    )

    val_dataset = ImageFolder(root=str(data_path / "val"), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"Classes: {val_dataset.classes}")

    # Evaluate accuracy
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, device)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Measure latency
    print("\nMeasuring latency (CPU, 100 runs)...")
    latency = measure_latency(model, device)

    print(f"\nLatency Results:")
    print(f"  Mean:   {latency['mean']:.2f} ms")
    print(f"  Median: {latency['median']:.2f} ms")
    print(f"  Std:    {latency['std']:.2f} ms")
    print(f"  Min:    {latency['min']:.2f} ms")
    print(f"  Max:    {latency['max']:.2f} ms")

    # Save results
    results = {
        "metrics": metrics,
        "latency": latency,
        "model": "convnextv2_tiny_baseline",
        "device": "cpu",
    }

    output_path = Path(
        "/home/dedmonds/repos/traina/temp-phase2-experiments/evaluation_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
