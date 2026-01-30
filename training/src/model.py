"""Model definition and utilities for ResNet50 classifier.

This module provides functions to create, save, and load the ResNet50 binary
classifier for cherry pit detection. The architecture matches the inference
code in cherry_detection/cherry_detection/ai_detector.py.
"""

from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


def create_classifier(
    architecture: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Create a classifier model (factory function).

    Args:
        architecture: Model architecture ('resnet50', 'mobilenet_v3_large', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: If True, freeze all layers except final classifier
        device: Device to move model to

    Returns:
        Configured PyTorch model
    """
    if architecture == "resnet50":
        return create_resnet50_classifier(
            num_classes, pretrained, freeze_backbone, device
        )

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if architecture == "mobilenet_v3_large":
        if pretrained:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            print("Loaded MobileNetV3-Large with ImageNet weights")
        else:
            model = mobilenet_v3_large(weights=None)
            print("Loaded MobileNetV3-Large without pretrained weights")

        if freeze_backbone:
            print("Freezing backbone layers")
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier
        # MobileNetV3 classifier is a Sequential, last layer is Linear
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        print(f"Replaced classifier head: Linear({num_ftrs}, {num_classes})")

    elif architecture == "efficientnet_b0":
        if pretrained:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            print("Loaded EfficientNet-B0 with ImageNet weights")
        else:
            model = efficientnet_b0(weights=None)
            print("Loaded EfficientNet-B0 without pretrained weights")

        if freeze_backbone:
            print("Freezing backbone layers")
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier
        # EfficientNet classifier is a Sequential, last layer is Linear
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        print(f"Replaced classifier head: Linear({num_ftrs}, {num_classes})")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)
    print(f"Model moved to device: {device}")
    return model


def create_resnet50_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Create ResNet50 binary classifier for cherry pit detection.

    Architecture matches ai_detector.py lines 148-156:
    - Load ResNet50 (pretrained on ImageNet)
    - Replace final fc layer: Linear(2048, num_classes)
    - Optionally freeze all layers except fc for faster training

    Args:
        num_classes: Number of output classes (default: 2 for clean/pit)
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: If True, freeze all layers except final fc layer
        device: Device to move model to ('cuda', 'cpu', or None for auto-detect)

    Returns:
        ResNet50 model configured for binary classification
    """
    # Load ResNet50
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print("Loaded ResNet50 with ImageNet pretrained weights")
    else:
        model = resnet50(weights=None)
        print("Loaded ResNet50 without pretrained weights")

    # Freeze backbone if requested
    if freeze_backbone:
        print("Freezing backbone layers (only training final fc layer)")
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer for binary classification
    # ResNet50 fc layer input features: 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final layer: Linear({num_ftrs}, {num_classes})")

    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model moved to device: {device}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint with training state.

    Saves full training state to enable resuming from checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        save_path: Path to save checkpoint file
        additional_info: Optional dict with extra info (e.g., metrics, config)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if additional_info:
        checkpoint.update(additional_info)

    # Create parent directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model checkpoint and optionally restore optimizer state.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to restore state into
        device: Device to load checkpoint to (None for auto-detect)

    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model weights loaded from: {checkpoint_path}")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state restored")

    # Return metadata
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
    }

    # Include any additional info from checkpoint
    for key, value in checkpoint.items():
        if key not in ["model_state_dict", "optimizer_state_dict", "epoch", "loss"]:
            metadata[key] = value

    print(f"Checkpoint loaded: epoch {metadata['epoch']}, loss {metadata['loss']:.4f}")

    return metadata


def save_model_weights_only(model: nn.Module, save_path: str) -> None:
    """Save only model weights (for deployment, not for resuming training).

    This creates a lightweight .pt file containing only the model state dict,
    matching the format used in inference (ai_detector.py).

    Args:
        model: Model to save
        save_path: Path to save weights file
    """
    # Create parent directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save only state dict (matches inference loading pattern)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved: {save_path}")


def load_model_weights_only(
    model: nn.Module, weights_path: str, device: Optional[str] = None
) -> nn.Module:
    """Load model weights from a state dict file.

    Args:
        model: Model architecture to load weights into
        weights_path: Path to weights file (.pt)
        device: Device to load weights to (None for auto-detect)

    Returns:
        Model with loaded weights

    Raises:
        FileNotFoundError: If weights file doesn't exist
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load weights
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    model.eval()  # Set to evaluation mode

    print(f"Model weights loaded from: {weights_path}")

    return model


if __name__ == "__main__":
    # Quick test of model creation
    print("Testing model creation...")

    # Test 1: Create model with pretrained weights
    model = create_resnet50_classifier(
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
        device="cpu",  # Use CPU for testing
    )

    print(f"\nModel architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Test 2: Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 128, 128)  # Batch of 1, RGB, 128x128
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output}")

    # Test 3: Test save/load
    print("\nTesting save/load...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/test_model.pt"
        save_model_weights_only(model, save_path)

        # Create new model and load weights
        model2 = create_resnet50_classifier(
            num_classes=2, pretrained=False, device="cpu"
        )
        model2 = load_model_weights_only(model2, save_path, device="cpu")

        # Verify outputs match
        output2 = model2(dummy_input)
        assert torch.allclose(output, output2), "Loaded model outputs don't match!"
        print("  Save/load test passed!")

    print("\nModel tests successful!")
