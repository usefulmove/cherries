#!/bin/bash
# Helper script to evaluate the unnormalized model against production baseline

# Set paths
REPO_ROOT="$(git rev-parse --show-toplevel)"
NEW_MODEL="$REPO_ROOT/training/experiments/resnet50_augmented_unnormalized/model_best.pt"
PROD_MODEL="$REPO_ROOT/cherry_system/cherry_detection/resource/cherry_classification.pt"
SCRIPT_PATH="$REPO_ROOT/training/scripts/compare_models.py"

# Check if new model exists
if [ ! -f "$NEW_MODEL" ]; then
    echo "Error: New model not found at $NEW_MODEL"
    echo "Please ensure the downloaded 'model_best.pt' is placed in that location."
    exit 1
fi

# Run comparison
echo "Running comparison..."
python "$SCRIPT_PATH" \
    --new-model "$NEW_MODEL" \
    --prod-model "$PROD_MODEL" \
    --unnormalized \
    --architecture resnet50

