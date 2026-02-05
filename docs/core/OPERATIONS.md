# Cherry Processing Operations

## Build & Test

### Building the Workspace
The system uses `colcon` for build management. Always run from the root of the workspace.

```bash
# Build entire workspace
colcon build --symlink-install

# Build specific package
colcon build --packages-select cherry_detection
```

### Running Tests
```bash
# Test entire workspace
colcon test

# Check test results
colcon test-result --all
```

## System Startup

### Standard Pipeline
To launch the full conveyor control system:

```bash
ros2 launch control_node conveyor_control_launch.py
```

### Robotic Sorter Integration
To launch the system with FANUC robot communication enabled:

```bash
ros2 launch fanuc_comms fanuc_launch.py
```

## Runtime Configuration

### Algorithm Selection
The `detection_server` can be configured at runtime to use different model variants:

```bash
# Get current algorithm
ros2 param get /detection_server algorithm

# Set algorithm (options: hdr_v1, hdr_v2, fasterRCNN-Mask_ResNet50_V1)
ros2 param set /detection_server algorithm fasterRCNN-Mask_ResNet50_V1
```

### Detection Thresholds
```bash
# Adjust pit detection confidence (0.0 - 1.0)
ros2 param set /detection_server pick_threshold 0.75

# Enable/Disable image saving for dataset collection
ros2 param set /control_node enable_image_save True
```

### Classification Thresholds (ai_detector.py:376-383)
```python
pit_mask = probs[:, 1].ge(.75)      # ≥75% confidence → pit
maybe_mask = probs[:, 1].ge(.5)     # ≥50% confidence → maybe (label 5)
clean_mask = probs[:, 0].ge(.5)     # ≥50% confidence → clean
```

## Reference: Classification Codes
The system uses the following integer codes for cherry classification:

| Code | Label | Description |
|------|-------|-------------|
| 1 | Clean | Good cherry; allowed to pass. |
| 2 | Pit | Defective; targeted for rejection/sorting. |
| 3 | Side | Edge of belt; position unreliable. |
| 5 | Maybe | Uncertain; yellow visual feedback. |

## Potential Classification Model Issues

### 1. Hard-coded Thresholds
- Fixed 75%/50% thresholds are not adaptive to:
  - Different lighting conditions
  - Cherry varieties
  - Camera sensor variations
- No calibration of softmax probabilities

### 2. Suboptimal Input Preprocessing (ai_detector.py:315-331)
- **Dilate-erode with 5×5 kernel**: Simple morphological ops may not be optimal
- **Bounding box crop**: Cuts off partial cherries at edges
- **CenterCrop to 128×128**: May lose important context/partial views
- **No padding handling**: Irregular crops get uniformly cropped

### 3. Limited Feature Utilization
- Only RGB pixels from masked region
- **Ignored**: 
  - Segmentation mask quality/score
  - Spatial context (size, position on conveyor)
  - Multi-scale features
  - Boundary/edge information

### 4. Model Architecture
- Generic ResNet50 without task-specific modifications
- No fine-tuning hyperparameters visible
- Single model - no ensemble or uncertainty estimation

### 5. Training Data Quality (inferred)
- No data augmentation evident in inference
- Hard to assess if training data matches deployment conditions

## Questions for Original Developer

### Training & Data Questions
1. **Dataset composition**: How many clean vs pit examples? What's the class balance?
2. **Data collection conditions**: What lighting, camera settings, cherry varieties were used?
3. **Data splits**: Train/val/test split ratios? Any temporal or spatial splits?
4. **Augmentation**: What augmentations were used during training?
5. **Label quality**: Manual or automated? Inter-annotator agreement?

### Model Architecture Questions
6. **Why ResNet50 specifically?** Were other architectures tested (EfficientNet, MobileNet, custom CNN)?
7. **Pre-training**: Used ImageNet pretrained weights or trained from scratch?
8. **Input size**: Why 128×128? Was this optimized through experimentation?
9. **Mask processing**: Why dilate-erode? Was morphology tuned?

### Training Process Questions
10. **Loss function**: Binary cross-entropy with logits? Class weighting?
11. **Optimizer**: Adam? SGD? What learning rate schedule?
12. **Training duration**: How many epochs? Convergence behavior?
13. **Hyperparameters**: Batch size, learning rate, weight decay, dropout?
14. **Regularization**: Any techniques used to prevent overfitting?

### Performance & Threshold Questions
15. **Threshold selection**: How were 75%/50% thresholds determined? ROC analysis?
16. **Metrics achieved**: What accuracy, precision, recall, F1 on validation set?
17. **Confusion matrix**: What are the common error types (clean→pit vs pit→clean)?
18. **False positive/negative tolerance**: What's the business cost of each error type?
19. **Latency requirements**: What's the acceptable inference time per frame?

### Operational Questions
20. **Deployment drift**: How does performance change with different lighting or cherry batches?
21. **Failure modes**: What types of cherries does it consistently misclassify?
22. **Maintenance**: How often is retraining needed? Any model decay observed?
23. **Calibration**: Are softmax probabilities well-calibrated?

### Improvement Questions
24. **Known limitations**: What did you identify as the biggest weaknesses?
25. **Unexplored ideas**: What improvements did you consider but not implement?
26. **Resource constraints**: Any compute/memory limitations for model changes?

## Suggested Improvement Approaches

### Quick Wins (Low Effort)
- **Threshold calibration**: Optimize thresholds based on cost of errors
- **Probability calibration**: Temperature scaling or Platt scaling
- **Ensemble**: Use multiple thresholds or models for uncertainty estimation

### Medium Effort
- **Better preprocessing**: Adaptive cropping, smart padding, multi-scale inputs
- **Feature engineering**: Include segmentation confidence, size, position as features
- **Architecture tuning**: Try EfficientNet-B0, MobileNetV3 for speed/accuracy trade-offs

### High Effort
- **End-to-end training**: Train segmentation+classification jointly
- **Domain-specific backbone**: Custom architecture for cherry texture analysis
- **Multi-task learning**: Predict additional attributes (size, quality, variety)
