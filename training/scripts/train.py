#!/usr/bin/env python3
"""Main training script for ResNet50 cherry classifier.

This script orchestrates the training loop, including:
- Loading config from YAML
- Creating data loaders
- Building model and optimizer
- Training and validation loops
- Checkpoint saving
- Metrics logging

Usage:
    python train.py --config configs/resnet50_baseline.yaml \\
                    --data-root ../cherry_classification/data \\
                    --output-dir ./outputs/baseline_run1
"""

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path to import training modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.model import (
    create_classifier,
    save_checkpoint,
    save_model_weights_only,
    load_checkpoint,
)
from src.metrics import (
    calculate_metrics,
    collect_predictions,
    log_metrics,
    print_metrics_summary,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    dry_run: bool = False,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        dry_run: If True, stop after a few batches

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f}"
            )

        # Dry run break
        if dry_run and batch_idx >= 2:
            print("Dry run: stopping training after 3 batches")
            break

    avg_loss = running_loss / (batch_idx + 1)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: list,
    dry_run: bool = False,
) -> tuple:
    """Validate the model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        class_names: List of class names
        dry_run: If True, stop after a few batches

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    # Collect predictions (custom logic for partial collection)
    if dry_run:
        # Mini collection for dry run
        y_true_list, y_pred_list, y_probs_list = [], [], []
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if i >= 3:
                    break
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                num_batches += 1

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                y_true_list.extend(labels.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())
                y_probs_list.extend(probs.cpu().numpy())

        avg_loss = running_loss / num_batches
        import numpy as np

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_probs = np.array(y_probs_list)
    else:
        # Full collection using collect_predictions utility
        y_true, y_pred, y_probs = collect_predictions(model, dataloader, device)

        # Calculate full loss
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)

    return avg_loss, metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ResNet50 cherry classifier")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to data directory (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (overrides config)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick smoke test (1 epoch, few batches)",
    )

    args = parser.parse_args()

    # Load config
    print(f"\nLoading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply dry-run settings
    if args.dry_run:
        print("\n" + "!" * 60)
        print("DRY RUN MODE ENABLED")
        print("!" * 60)
        config["training"]["epochs"] = 1
        config["logging"]["print_every"] = 1
        config["checkpointing"]["save_every"] = 1

    # Override config with command line arguments
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir

    # Create output directory
    output_dir = Path(config["checkpointing"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config to output directory for reproducibility
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Set device
    device_config = config["training"]["device"]
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    print(f"Using device: {device}")

    # Create data loaders
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_loader, val_loader = get_dataloaders(
        data_root=config["data"]["root"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        input_size=config["data"]["input_size"],
        augmentation=config["data"]["augmentation"],
        normalize=config["data"]["normalize"],
    )

    # Get class names from dataset
    class_names = train_loader.dataset.classes
    print(f"Class names: {class_names}")

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    architecture = config["model"].get("architecture", "resnet50")
    print(f"Architecture: {architecture}")

    model = create_classifier(
        architecture=architecture,
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        device=device,
    )

    # Create optimizer
    optimizer_name = config["training"]["optimizer"].lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']}")

    # Create scheduler
    scheduler = None
    if config["training"].get("use_scheduler", False):
        scheduler_type = config["training"].get("scheduler_type", "step")
        print(f"Using scheduler: {scheduler_type}")

        if scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["training"]["scheduler_step_size"],
                gamma=config["training"]["scheduler_gamma"],
            )
        elif scheduler_type == "cosine":
            # Default T_max to epochs if not specified
            t_max = config["training"].get(
                "scheduler_t_max", config["training"]["epochs"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=0,  # Decay to 0
            )
        else:
            print(
                f"Warning: Unknown scheduler type {scheduler_type}, skipping scheduler"
            )

    # Create loss function with optional label smoothing
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
    else:
        print("Using standard CrossEntropyLoss (no label smoothing)")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint_data = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = checkpoint_data["epoch"] + 1
        best_val_acc = checkpoint_data.get("best_val_acc", 0.0)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training for {config['training']['epochs']} epochs")
    print(f"Starting from epoch {start_epoch}")

    log_file = output_dir / config["logging"]["log_file"]

    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start_time = datetime.now()

        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{config['training']['epochs']}")
        print(f"{'=' * 60}")

        # Training phase
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch + 1,
            dry_run=args.dry_run,
        )
        print(f"\nTraining Loss: {train_loss:.4f}")

        # Validation phase
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            class_names,
            dry_run=args.dry_run,
        )
        print(f"Validation Loss: {val_loss:.4f}")

        # Step scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Stepped scheduler. New LR: {current_lr:.6f}")

        # Print validation metrics
        print_metrics_summary(val_metrics, epoch + 1, "val")

        # Log metrics
        log_metrics(
            metrics={"loss": train_loss},
            epoch=epoch + 1,
            phase="train",
            output_file=str(log_file),
            experiment_name=config["experiment"]["name"],
        )

        val_metrics["loss"] = val_loss
        log_metrics(
            metrics=val_metrics,
            epoch=epoch + 1,
            phase="val",
            output_file=str(log_file),
            experiment_name=config["experiment"]["name"],
            additional_info={"learning_rate": config["training"]["learning_rate"]},
        )

        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_every"] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                str(checkpoint_path),
                additional_info={
                    "val_accuracy": val_metrics["accuracy"],
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
            )

        # Save best model
        val_acc = val_metrics["accuracy"]
        if config["checkpointing"]["save_best"] and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / "model_best.pt"
            save_model_weights_only(model, str(best_model_path))
            print(f"\nNew best model saved! Validation accuracy: {best_val_acc:.4f}")

        # Print epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        print(f"\nEpoch completed in {epoch_time:.1f} seconds")

    # Save final model
    final_model_path = output_dir / "model_final.pt"
    save_model_weights_only(model, str(final_model_path))
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
