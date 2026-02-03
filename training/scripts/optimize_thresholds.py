#!/usr/bin/env python3
"""Threshold optimization script for cherry pit detection.

This script analyzes probability distributions and finds optimal decision
thresholds that minimize missed pits (false negatives) while maintaining
acceptable false positive rates.

Usage:
    python optimize_thresholds.py \\
        --model-path experiments/resnet50_augmented_unnormalized/model_best_fixed.pt \\
        --data-root ../cherry_classification/data \\
        --architecture resnet50 \\
        --output-dir ./threshold_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_classifier, load_model_weights_only
from src.data import get_dataloaders


def collect_probabilities(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect true labels and predicted probabilities.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run inference on

    Returns:
        Tuple of (y_true, y_probs) where y_probs[:, 1] is pit probability
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)

    return y_true, y_probs


def analyze_threshold_range(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_range: List[float],
    target_class: int = 1,
) -> Dict:
    """Analyze metrics across threshold range.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        threshold_range: List of thresholds to test
        target_class: Class to analyze (1 = pit)

    Returns:
        Dictionary with metrics for each threshold
    """
    results = []

    for threshold in threshold_range:
        y_pred = (y_probs[:, target_class] >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    return results


def plot_threshold_analysis(
    results: List[Dict],
    output_path: Path,
    target_recall: float = 0.95,
):
    """Plot threshold analysis results.

    Args:
        results: List of result dictionaries from analyze_threshold_range
        output_path: Path to save plot
        target_recall: Target recall line to draw
    """
    thresholds = [r["threshold"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision, Recall, F1
    ax1.plot(thresholds, precisions, "b-", label="Precision", linewidth=2)
    ax1.plot(thresholds, recalls, "r-", label="Recall (Pit)", linewidth=2)
    ax1.plot(thresholds, f1s, "g-", label="F1 Score", linewidth=2)
    ax1.axhline(
        y=target_recall,
        color="orange",
        linestyle="--",
        label=f"Target Recall={target_recall}",
    )
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision, Recall, F1 vs Threshold", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy and False Negatives
    ax2_twin = ax2.twinx()

    ax2.plot(thresholds, accuracies, "purple", label="Accuracy", linewidth=2)
    false_negatives = [r["fn"] for r in results]
    ax2_twin.plot(
        thresholds,
        false_negatives,
        "red",
        linestyle="--",
        label="False Negatives (Missed Pits)",
        linewidth=2,
    )

    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12, color="purple")
    ax2_twin.set_ylabel("False Negatives", fontsize=12, color="red")
    ax2.set_title("Accuracy and Missed Pits vs Threshold", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Threshold analysis plot saved: {output_path}")
    plt.close()


def plot_probability_distribution(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
    class_names: List[str] = ["Clean", "Pit"],
):
    """Plot probability distributions for each class.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        output_path: Path to save plot
        class_names: Names of classes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        # Get probabilities for this class
        class_mask = y_true == class_idx
        correct_probs = y_probs[class_mask, class_idx]
        incorrect_probs = y_probs[~class_mask, class_idx]

        # Plot histograms
        ax.hist(
            correct_probs, bins=50, alpha=0.7, color="green", label=f"True {class_name}"
        )
        ax.hist(
            incorrect_probs,
            bins=50,
            alpha=0.7,
            color="red",
            label=f"False {class_name}",
        )

        ax.set_xlabel(f"Predicted Probability ({class_name})", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{class_name} Probability Distribution", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Probability distribution plot saved: {output_path}")
    plt.close()


def find_optimal_threshold(
    results: List[Dict],
    min_recall: float = 0.95,
    metric: str = "f1",
) -> Dict:
    """Find optimal threshold based on constraints.

    Args:
        results: List of result dictionaries
        min_recall: Minimum acceptable recall (for pit detection)
        metric: Metric to optimize ('f1', 'accuracy', 'precision')

    Returns:
        Best result dictionary
    """
    # Filter to meet minimum recall constraint
    valid_results = [r for r in results if r["recall"] >= min_recall]

    if not valid_results:
        print(f"Warning: No thresholds meet min_recall={min_recall}")
        print("Falling back to threshold with best recall")
        return max(results, key=lambda x: x["recall"])

    # Find best by metric
    best = max(valid_results, key=lambda x: x[metric])
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Optimize classification thresholds for pit detection"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to data directory (with val/ subdirectory)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        help="Model architecture (resnet18, resnet50, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./threshold_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Minimum acceptable recall for pit detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = create_classifier(
        architecture=args.architecture,
        num_classes=2,
        pretrained=False,
        device=device,
    )
    model = load_model_weights_only(model, args.model_path, device=device)

    # Load validation data
    print(f"\nLoading validation data from: {args.data_root}")
    _, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        num_workers=4,
        input_size=128,
        augmentation=False,  # No augmentation for validation
        normalize=False,  # Match production (unnormalized)
    )

    print(f"Validation samples: {len(val_loader.dataset)}")

    # Collect probabilities
    print("\nCollecting predictions...")
    y_true, y_probs = collect_probabilities(model, val_loader, device)
    print(f"Collected {len(y_true)} samples")

    # Plot probability distributions
    print("\nPlotting probability distributions...")
    plot_probability_distribution(
        y_true, y_probs, output_dir / "probability_distributions.png"
    )

    # Analyze threshold range
    print(f"\nAnalyzing thresholds (min_recall={args.min_recall})...")
    threshold_range = np.linspace(0.1, 0.95, 50)
    results = analyze_threshold_range(y_true, y_probs, threshold_range, target_class=1)

    # Plot threshold analysis
    plot_threshold_analysis(
        results,
        output_dir / "threshold_analysis.png",
        target_recall=args.min_recall,
    )

    # Find optimal thresholds
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 60)

    for metric in ["f1", "accuracy", "precision"]:
        optimal = find_optimal_threshold(
            results, min_recall=args.min_recall, metric=metric
        )
        print(f"\nOptimizing for {metric.upper()} (min_recall={args.min_recall}):")
        print(f"  Threshold: {optimal['threshold']:.3f}")
        print(f"  Accuracy:  {optimal['accuracy']:.4f}")
        print(f"  Precision: {optimal['precision']:.4f}")
        print(f"  Recall:    {optimal['recall']:.4f}")
        print(f"  F1 Score:  {optimal['f1']:.4f}")
        print(f"  Missed Pits (FN): {optimal['fn']}")
        print(f"  False Alarms (FP): {optimal['fp']}")

    # Save detailed results
    results_file = output_dir / "threshold_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved: {results_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
