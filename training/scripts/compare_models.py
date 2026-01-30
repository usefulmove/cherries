import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import argparse
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.metrics import calculate_metrics
from src.model import create_classifier


def load_prod_model(path, device):
    """Load production model (ResNet50 state dict)."""
    # Production model is always ResNet50
    model = create_classifier(
        architecture="resnet50",
        num_classes=2,
        pretrained=False,  # We load weights manually
        device=device,
    )

    # Load state dict
    print(f"Loading weights from: {path}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_new_model(path, architecture, device):
    """Load new model with specified architecture."""
    model = create_classifier(
        architecture=architecture,
        num_classes=2,
        pretrained=False,  # We load weights manually
        device=device,
    )

    # Load state dict
    print(f"Loading weights from: {path}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def evaluate(model, dataloader, device, desc):
    """Run inference and calculate metrics."""
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"\nEvaluating: {desc}")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get probabilities and predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Class names from dataset (assumed [clean, pit] based on alphabetical order)
    class_names = ["cherry_clean", "cherry_pit"]
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)
    return metrics


def print_metrics(metrics, title):
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    if "per_class" in metrics:
        print("Per-Class Metrics:")
        for cls, scores in metrics["per_class"].items():
            print(
                f"  {cls}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['f1']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Compare Production vs New Model")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../../cherry_classification/data",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--new-model", type=str, required=True, help="Path to new model .pt"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        help="Architecture of new model (resnet50, mobilenet_v3_large, efficientnet_b0)",
    )
    parser.add_argument(
        "--prod-model", type=str, required=True, help="Path to production model .pt"
    )
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Specify if the NEW model expects unnormalized data (0-255). Default is False (ImageNet norm).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")

    # 1. Load DataLoaders
    print("\nLoading datasets...")
    # Normalized (for New Model & Prod Model fixed)
    _, val_loader_norm = get_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        normalize=True,
        augmentation=False,
        num_workers=2,
    )

    # Raw (for Prod Model baseline)
    _, val_loader_raw = get_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        normalize=False,
        augmentation=False,
        num_workers=2,
    )

    # 2. Load Models
    print("\nLoading models...")
    new_model = load_new_model(args.new_model, args.architecture, device)
    prod_model = load_prod_model(args.prod_model, device)

    # 3. Evaluate
    # A. Production Model (Baseline - No Normalization)
    metrics_prod_raw = evaluate(
        prod_model,
        val_loader_raw,
        device,
        "Production Model (Current System - No Norm)",
    )

    # B. Production Model (With Normalization - Hypothetical Fix)
    metrics_prod_norm = evaluate(
        prod_model, val_loader_norm, device, "Production Model (With Norm Fix)"
    )

    # C. New Model
    if args.unnormalized:
        # If new model is unnormalized, evaluate it on RAW data
        metrics_new = evaluate(
            new_model,
            val_loader_raw,
            device,
            f"New Model ({args.architecture} - Unnormalized)",
        )
    else:
        # Default: Evaluate on NORMALIZED data
        metrics_new = evaluate(
            new_model,
            val_loader_norm,
            device,
            f"New Model ({args.architecture} - Normalized)",
        )

    # 4. Report
    print_metrics(metrics_prod_raw, "Production Model (Baseline)")
    print_metrics(metrics_prod_norm, "Production Model (If Fixed)")
    print_metrics(metrics_new, "New Trained Model")

    # Summary Table
    print("\n" + "=" * 95)
    print(
        f"{'Metric':<15} | {'Prod (Current)':<22} | {'Prod (Fixed)':<22} | {'New Model':<22}"
    )
    print("-" * 95)
    print(
        f"{'Accuracy':<15} | {metrics_prod_raw['accuracy']:<22.4f} | {metrics_prod_norm['accuracy']:<22.4f} | {metrics_new['accuracy']:<22.4f}"
    )
    print(
        f"{'Precision':<15} | {metrics_prod_raw['precision']:<22.4f} | {metrics_prod_norm['precision']:<22.4f} | {metrics_new['precision']:<22.4f}"
    )
    print(
        f"{'Recall':<15} | {metrics_prod_raw['recall']:<22.4f} | {metrics_prod_norm['recall']:<22.4f} | {metrics_new['recall']:<22.4f}"
    )
    print(
        f"{'F1 Score':<15} | {metrics_prod_raw['f1']:<22.4f} | {metrics_prod_norm['f1']:<22.4f} | {metrics_new['f1']:<22.4f}"
    )
    print("=" * 95)


if __name__ == "__main__":
    main()
