import os
import shutil
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

# --- Configuration ---
TEMP_DIR = Path("temp_2stage_test")
DATA_DIR = TEMP_DIR / "data"
STAGE1_DIR = DATA_DIR / "stage1"
STAGE2_DIR = DATA_DIR / "stage2"
MODEL_DIR = TEMP_DIR / "models"
CLASSES = ["clean", "pit"]
IMG_SIZE = 64
BATCH_SIZE = 4


def setup_mock_data():
    """Create dummy dataset with random images."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    STAGE1_DIR.mkdir(parents=True)
    MODEL_DIR.mkdir(parents=True)

    print(f"Creating mock data in {STAGE1_DIR}...")

    for cls in CLASSES:
        (STAGE1_DIR / cls).mkdir()
        # Create 10 dummy images per class
        for i in range(10):
            img = Image.new(
                "RGB",
                (IMG_SIZE, IMG_SIZE),
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
            img.save(STAGE1_DIR / cls / f"{cls}_{i}.jpg")


def get_model(num_classes=2):
    """Get a simple ResNet-like model for testing."""
    # Using a tiny model for speed
    model = torch.nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


def stage1_pipeline():
    print("\n--- STAGE 1: Binary Training ---")

    # Data Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(STAGE1_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model Setup
    model = get_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train
    print("Training Stage 1 (Mock)...")
    model = train_one_epoch(model, dataloader, criterion, optimizer)

    # Save
    torch.save(model.state_dict(), MODEL_DIR / "stage1.pt")
    print("Stage 1 model saved.")
    return model


def mining_step(model):
    print("\n--- MINING STEP: Identifying 'Maybe' ---")

    # Copy data to Stage 2 folder first
    if STAGE2_DIR.exists():
        shutil.rmtree(STAGE2_DIR)
    shutil.copytree(STAGE1_DIR, STAGE2_DIR)
    (STAGE2_DIR / "maybe").mkdir()

    # Load Data (No Shuffle to track indices)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(STAGE1_DIR, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False
    )  # Batch 1 for simplicity

    model.eval()
    moves = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):
            output = model(img)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            confidence = probs.max().item()

            # Logic: Move if low confidence OR wrong prediction
            # For this test, let's force moves for even indices to verify logic
            is_hard = i % 2 == 0

            if is_hard:
                # Get source path
                sample_path, _ = dataset.samples[i]
                filename = Path(sample_path).name
                src_cls = CLASSES[label.item()]

                # Determine destination
                dest = STAGE2_DIR / "maybe" / filename
                src = STAGE2_DIR / src_cls / filename

                # Move file
                shutil.move(str(src), str(dest))
                moves += 1

    print(f"Moved {moves} images to 'maybe' class.")

    # Clean up empty folders if needed (not strictly necessary for test)
    return moves


def stage2_pipeline():
    print("\n--- STAGE 2: 3-Class Fine-Tuning ---")

    # Verify dataset structure
    classes = [d.name for d in STAGE2_DIR.iterdir() if d.is_dir()]
    print(f"Stage 2 Classes: {sorted(classes)}")
    assert "maybe" in classes

    # Data Setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(STAGE2_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Stage 1 weights
    print("Loading Stage 1 weights...")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_DIR / "stage1.pt"))

    # Modify Head
    print("Modifying head for 3 classes...")
    in_features = model[-1].in_features
    model[-1] = nn.Linear(in_features, 3)  # Clean, Pit, Maybe

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print("Training Stage 2 (Mock)...")
    model = train_one_epoch(model, dataloader, criterion, optimizer)

    # Save
    torch.save(model.state_dict(), MODEL_DIR / "stage2.pt")
    print("Stage 2 model saved.")


def main():
    try:
        setup_mock_data()
        model = stage1_pipeline()
        mining_step(model)
        stage2_pipeline()
        print("\n✅ VERIFICATION SUCCESSFUL: 2-Stage Logic Works!")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        raise
    finally:
        # Cleanup
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main()
