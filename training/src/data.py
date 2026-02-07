"""Data loading and preprocessing for cherry classification.

This module provides PyTorch Dataset and DataLoader utilities for loading
cherry images from the dataset directory structure:
    data/
    ├── train/
    │   ├── cherry_clean/
    │   └── cherry_pit/
    └── val/
        ├── cherry_clean/
        └── cherry_pit/
"""

from typing import Tuple, Optional
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(
    input_size: int = 128, augmentation: bool = False, normalize: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms.

    Args:
        input_size: Size of the input image (default: 128 to match inference)
        augmentation: Whether to apply data augmentation to training set
        normalize: Whether to apply ImageNet normalization

    Returns:
        Tuple of (train_transform, val_transform)
    """
    # ImageNet normalization parameters
    # CRITICAL: Must match inference preprocessing for consistency
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Base transforms (always applied)
    base_transforms = [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]

    if normalize:
        base_transforms.append(
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        )

    # Training transforms (with optional augmentation)
    train_transform_list = []
    if augmentation:
        # Standard geometric augmentations
        train_transform_list.extend(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=180),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
            ]
        )

        # Enhanced conveyor belt realism augmentations (Phase 2)
        # Motion blur simulates conveyor movement
        train_transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3
            )
        )

        # Stronger photometric distortions for lighting/shadow simulation
        train_transform_list.append(
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )
        )
    train_transform_list.extend(base_transforms)

    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose(base_transforms)

    return train_transform, val_transform


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 128,
    augmentation: bool = False,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Expects directory structure:
        data_root/
        ├── train/
        │   ├── cherry_clean/  (label 0 by alphabetical order)
        │   └── cherry_pit/    (label 1 by alphabetical order)
        └── val/
            ├── cherry_clean/
            └── cherry_pit/

    Args:
        data_root: Path to the data directory (must contain train/ and val/)
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        input_size: Size of input images (default: 128)
        augmentation: Whether to apply data augmentation to training set
        normalize: Whether to apply ImageNet normalization

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        FileNotFoundError: If data_root, train/, or val/ directories don't exist
    """
    data_path = Path(data_root)

    # Validate directory structure
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            f"Please clone the dataset: "
            f"git clone https://github.com/weshavener/cherry_classification.git"
        )

    train_path = data_path / "train"
    val_path = data_path / "val"

    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")

    # Get transforms
    train_transform, val_transform = get_transforms(
        input_size=input_size, augmentation=augmentation, normalize=normalize
    )

    # Create datasets
    # ImageFolder automatically assigns labels alphabetically:
    # cherry_clean = 0, cherry_pit = 1
    train_dataset = datasets.ImageFolder(
        root=str(train_path), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(root=str(val_path), transform=val_transform)

    # Print dataset info
    print(f"\nDataset loaded from: {data_root}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Class distribution (train): {_get_class_distribution(train_dataset)}")
    print(f"Class distribution (val): {_get_class_distribution(val_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU->GPU transfer
        drop_last=True,  # Drop incomplete batch for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def _get_class_distribution(dataset: datasets.ImageFolder) -> dict:
    """Get the distribution of classes in the dataset.

    Args:
        dataset: ImageFolder dataset

    Returns:
        Dictionary mapping class names to counts
    """
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts


if __name__ == "__main__":
    # Quick test of data loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data.py <path_to_data_root>")
        print("Example: python data.py ../cherry_classification/data")
        sys.exit(1)

    data_root = sys.argv[1]

    print("Testing data loading...")
    train_loader, val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        input_size=128,
        augmentation=False,
    )

    # Load one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print("\nData loading test successful!")
