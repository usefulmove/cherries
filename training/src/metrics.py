"""Metrics calculation and logging utilities.

This module provides functions for calculating classification metrics
and logging them to JSON files for experiment tracking.
"""

from typing import Dict, List, Any
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Calculate classification metrics.

    Args:
        y_true: True labels (shape: [N])
        y_pred: Predicted labels (shape: [N])
        y_probs: Predicted probabilities (shape: [N, num_classes])
        class_names: Optional list of class names for labeling

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Macro-averaged precision
        - recall: Macro-averaged recall
        - f1: Macro-averaged F1 score
        - confusion_matrix: Confusion matrix as nested list
        - per_class_metrics: Dict with per-class precision/recall/f1
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(y_probs.shape[1])]

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def collect_predictions(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> tuple:
    """Collect predictions from a model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run inference on

    Returns:
        Tuple of (y_true, y_pred, y_probs) as numpy arrays
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get probabilities and predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


def log_metrics(
    metrics: Dict[str, Any],
    epoch: int,
    phase: str,
    output_file: str,
    experiment_name: str = None,
    additional_info: Dict[str, Any] = None,
) -> None:
    """Log metrics to a JSON file.

    Appends a single line of JSON to the log file (JSON Lines format).
    Each line is a complete JSON object that can be parsed independently.

    Args:
        metrics: Dictionary of metrics to log
        epoch: Current epoch number
        phase: Training phase ('train' or 'val')
        output_file: Path to output JSON file
        experiment_name: Optional experiment name
        additional_info: Optional additional info to log (e.g., learning rate)
    """
    # Create parent directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Build log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "phase": phase,
    }

    if experiment_name:
        log_entry["experiment"] = experiment_name

    # Add metrics with phase prefix
    for key, value in metrics.items():
        log_entry[f"{phase}_{key}"] = value

    # Add additional info
    if additional_info:
        log_entry.update(additional_info)

    # Append to file (JSON Lines format)
    with open(output_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")


def print_metrics_summary(metrics: Dict[str, Any], epoch: int, phase: str) -> None:
    """Print a formatted summary of metrics to console.

    Args:
        metrics: Dictionary of metrics to print
        epoch: Current epoch number
        phase: Training phase ('train' or 'val')
    """
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch} - {phase.upper()}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    # Print confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    print(f"\nConfusion Matrix:")
    print(cm)

    # Print per-class metrics if available
    if "per_class_metrics" in metrics:
        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1:        {class_metrics['f1']:.4f}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Quick test of metrics calculation
    print("Testing metrics calculation...")

    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 2

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize

    # Calculate metrics
    class_names = ["cherry_clean", "cherry_pit"]
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)

    # Print summary
    print_metrics_summary(metrics, epoch=1, phase="test")

    # Test logging
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = f"{tmpdir}/test_metrics.json"

        log_metrics(
            metrics=metrics,
            epoch=1,
            phase="train",
            output_file=log_file,
            experiment_name="test_experiment",
            additional_info={"learning_rate": 0.001},
        )

        # Read and verify
        with open(log_file, "r") as f:
            logged_data = json.loads(f.readline())
            print(f"Logged data keys: {list(logged_data.keys())}")
            print(f"Train accuracy logged: {logged_data['train_accuracy']:.4f}")

    print("\nMetrics tests successful!")
