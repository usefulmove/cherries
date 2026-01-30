#!/usr/bin/env python3
"""Plot training metrics from JSON log file.

This utility reads the metrics.json file (JSON Lines format) and generates
training curves (loss and accuracy) as PNG images.

Usage:
    python plot_metrics.py outputs/baseline/metrics.json
    python plot_metrics.py outputs/baseline/metrics.json --output curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_file: str) -> dict:
    """Load metrics from JSON Lines log file.

    Args:
        log_file: Path to metrics.json file

    Returns:
        Dictionary with 'train' and 'val' lists of metric dicts
    """
    train_metrics = []
    val_metrics = []

    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                phase = data.get("phase", "")
                if phase == "train":
                    train_metrics.append(data)
                elif phase == "val":
                    val_metrics.append(data)

    return {"train": train_metrics, "val": val_metrics}


def plot_training_curves(metrics: dict, output_path: str = None):
    """Plot training and validation curves.

    Args:
        metrics: Dictionary with 'train' and 'val' metrics
        output_path: Optional path to save figure (default: show plot)
    """
    train_data = metrics["train"]
    val_data = metrics["val"]

    if not train_data or not val_data:
        print("No metrics data found!")
        return

    # Extract epochs and metrics
    train_epochs = [d["epoch"] for d in train_data]
    train_loss = [d.get("train_loss", 0) for d in train_data]

    val_epochs = [d["epoch"] for d in val_data]
    val_loss = [d.get("val_loss", 0) for d in val_data]
    val_acc = [d.get("val_accuracy", 0) for d in val_data]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1.plot(train_epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax1.plot(val_epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation accuracy
    ax2.plot(val_epochs, val_acc, "g-", label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])

    # Add text box with best metrics
    if val_acc:
        best_acc = max(val_acc)
        best_epoch = val_epochs[val_acc.index(best_acc)]
        textstr = f"Best Val Acc: {best_acc:.4f}\n(Epoch {best_epoch})"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax2.text(
            0.05,
            0.95,
            textstr,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary(metrics: dict):
    """Print summary statistics from metrics.

    Args:
        metrics: Dictionary with 'train' and 'val' metrics
    """
    val_data = metrics["val"]

    if not val_data:
        print("No validation metrics found!")
        return

    # Extract metrics
    val_acc = [d.get("val_accuracy", 0) for d in val_data]
    val_loss = [d.get("val_loss", 0) for d in val_data]

    # Calculate stats
    best_acc = max(val_acc)
    best_epoch = val_data[val_acc.index(best_acc)]["epoch"]
    final_acc = val_acc[-1]
    final_loss = val_loss[-1]

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total epochs: {len(val_data)}")
    print(f"\nBest validation accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Final validation accuracy: {final_acc:.4f}")
    print(f"Final validation loss: {final_loss:.4f}")

    # Calculate improvement
    if len(val_acc) > 1:
        improvement = final_acc - val_acc[0]
        print(f"Improvement from epoch 1: {improvement:+.4f}")

    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from JSON log file"
    )
    parser.add_argument("log_file", type=str, help="Path to metrics.json file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: show plot)",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip printing summary statistics"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.log_file).exists():
        print(f"Error: File not found: {args.log_file}")
        return

    # Load metrics
    print(f"Loading metrics from: {args.log_file}")
    metrics = load_metrics(args.log_file)

    # Print summary
    if not args.no_summary:
        print_summary(metrics)

    # Generate default output path if not specified
    output_path = args.output
    if output_path is None:
        log_dir = Path(args.log_file).parent
        output_path = log_dir / "training_curves.png"

    # Plot
    plot_training_curves(metrics, str(output_path))


if __name__ == "__main__":
    main()
