# Discussion: Cherry Processing System
**Date:** February 5, 2026  

---

## 1. Current State (Classification)

### Model Evaluation
Working evaluator built from `Copy_of_cherry_classification_evaluation.ipynb`. Validates models against dataset from `https://github.com/weshavener/cherry_classification` repo. Can load and test trained models files to verify accuracy on validation set.

### Model Training
Working training pipeline in `training/scripts/train.py` and Colab notebook (`colab_optimization_experiments.ipynb`). Supports different backbones, configurable augmentation, unnormalized training to match production.

### Experiments

Explored multiple optimization strategies to improve classification accuracy:

**Approaches Tested:**
- **Data augmentation** (rotation, affine transforms, color jitter) - improved accuracy by ~1%
- **Training normalization** - discovered original mismatch between training (normalized) and inference (unnormalized); resolved by training on 0-255 range
- **Architecture alternatives** - tested MobileNetV3, EfficientNet B0, ResNet18; ResNet50 remains best for accuracy
- **Differential learning rates** - freezing early layers and fine-tuning deeper layers (inconclusive results)

**Current Best Result:** ResNet50 with augmentation, unnormalized training achieves **94.05% accuracy** (vs 92.99% production baseline)

**Alternative Option:** ResNet18 trades 1% accuracy for 40% speed improvement and 50% model size reduction

### Current Model Candidates

| Model | Accuracy | Size | Latency | Status |
|-------|----------|------|---------|--------|
| Production Baseline | 92.99% | 90MB | ~16ms | Currently deployed |
| **ResNet50 Best** | **94.05%** | 90MB | ~16.7ms | **???** |
| ResNet18 Option | 91.92% | 43MB | ~8-10ms | Speed alternative |

### Infrastructure Status

- **Evaluation:** Working pipeline to test models against validation data
- **Training:** Colab notebook with skip-flag pattern for systematic experiments
- **Optimization:** Scripts ready for threshold tuning and latency benchmarking
- **Models:** 94.05% ResNet50 ready for deployment consideration

---

## 2. Training Data

### Current Understanding

From the evaluation notebook (`Copy_of_cherry_classification_evaluation.ipynb`):
- **Data Source:** `https://github.com/weshavener/cherry_classification` (cloned in notebook)
- **Date:** Data collected on 11/2/2022
- **Structure:** GitHub repo contains train/val split
- **Models referenced:**
  - `classification_11_2022_all_data_adam.pt`
  - `classification_11_2022_all_data_sgd.pt`
  - `classification_old.pt`

### Discussion

- Is `https://github.com/weshavener/cherry_classification` the original training data?

---

## 3. Training Scripts & Hyperparameters

### Current Understanding

I have the **evaluation notebook** but not the original **training scripts**. The evaluation notebook shows:
- ResNet50 architecture with custom FC layer (2 classes)
- 128×128 CenterCrop preprocessing

### Discussion

**Original training configuration**

**Questions:**
- Where are the original training scripts (hyperparameters (LR, batch size, epochs), etc.)?
- Which optimizer was used (the notebook shows both Adam and SGD models)?
- What data augmentation was originally applied?

---

## 4. Deployment Process

### Discussion

**Questions:**
- What is the deployment process for model updates?
- Is there version control for model weights?

---

## 5. Threshold Optimization & Business Requirements

### Current Understanding

**Current Thresholds:**
- pit ≥ 0.75 → label 2 (pit)
- maybe ≥ 0.5 → label 5 (uncertain)
- clean ≥ 0.5 → label 1 (clean)

**Maybe Category Handling - VERIFIED:**

The "maybe" category (label 5) is **visually highlighted for manual worker review**:

1. **Classification Logic** (`ai_detector.py:246, 376-383`):
   - Label 5 assigned when pit probability ≥ 0.5 but < 0.75
   - Locations array: `['none', 'cherry_clean', 'cherry_pit', 'side', 'top/bot', 'maybe']`

2. **Detection Visualization** (`ai_detector.py:480-483`):
   - Clean: **Lime green** bounding boxes
   - Pit: **Red** bounding boxes
   - **Maybe: Yellow bounding boxes** (for debug images)

3. **Belt Projection** (`helper.cpp:69, 115-134`):
   - **Type 5 = Yellow circles** projected onto physical belt
   - Workers can visually identify "maybe" cherries for manual inspection
   - Code: `circleBrush_maybe = QBrush(Qt::yellow)`

**Result:** "Maybe" cherries are **highlighted in yellow on the projection system** for workers to manually review. This confirms the system already has a manual review pathway for uncertain predictions.

### Discussion

**Optimize decision thresholds?**

**Questions:**
- What is the impact of false negatives (missed pits)?
- What is the cost of false positives (clean cherries marked as pits)?
- How does the "maybe" category work in practice?
