## **System Overview**

The cherry processing system uses a **two-stage deep learning pipeline**:

### **Stage 1: Segmentation (Mask R-CNN)**
- **Model**: `torchvision.models.detection.maskrcnn_resnet50_fpn`
- **Classes**: 2 (background=0, cherry_matter=1)
- **Input**: Color image (2464×500 pixels after cropping)
- **Output**: Bounding boxes + binary masks for each detected cherry
- **Weights**: `cherry_segmentation.pt` in `control_node/resource/`

### **Stage 2: Classification (ResNet50)**
- **Model**: ResNet50 with modified final layer (2 outputs)
- **Classes**: Binary (clean_cherry=0, pit=1)
- **Input**: 128×128 RGB patches extracted from segmented regions
- **Processing**: 
  - Apply segmentation mask to crop cherry from background
  - Dilate-erode mask with 5×5 kernel
  - CenterCrop to 128×128
- **Weights**: `cherry_classification.pt` in `control_node/resource/`

### **Node Interaction & Deployment Architecture**

The system employs a service-based architecture where detection is decoupled from control:

1.  **Client (`control_node`)**:
    - Orchestrates the pipeline.
    - Acquires images but does *not* process them locally.
    - Sends images to the `cherry_detection` node via the `detection_server/detect` service.
    
2.  **Server (`cherry_detection`)**:
    - Listens for detection requests.
    - Loads the AI models and performs inference.
    - Returns coordinates and types of detected cherries.

### **Legacy Artifacts & Configuration**

**Important Note on Model Loading:**
Due to a known configuration bug in `cherry_detection/detector.py`, the active detection node loads model weights from the `control_node` package instead of its own package.

- **Active Weights:** Located in `control_node/resource/` (loaded by `cherry_detection` at runtime).
- **Inactive Weights:** Located in `cherry_detection/resource/` (never loaded, dead code).
- **Legacy Code:** `control_node` contains a full copy of the detection source code (`detector.py`, `ai_detector.py`) which is not used at runtime.

See `docs/known-issues.md` for cleanup plans.

### **Classification Thresholds** (ai_detector.py:376-383)
```python
pit_mask = probs[:, 1].ge(.75)      # ≥75% confidence → pit
maybe_mask = probs[:, 1].ge(.5)     # ≥50% confidence → maybe (label 5)
clean_mask = probs[:, 0].ge(.5)     # ≥50% confidence → clean
```

### **Additional Labels**
- Label 3: Side cherries (position-based: x < 170 or x > 2244)
- Label 5: Maybe (50-75% pit probability)

---

## **Potential Classification Model Issues**

### **1. Hard-coded Thresholds**
- Fixed 75%/50% thresholds are not adaptive to:
  - Different lighting conditions
  - Cherry varieties
  - Camera sensor variations
- No calibration of softmax probabilities

### **2. Suboptimal Input Preprocessing** (ai_detector.py:315-331)
- **Dilate-erode with 5×5 kernel**: Simple morphological ops may not be optimal
- **Bounding box crop**: Cuts off partial cherries at edges
- **CenterCrop to 128×128**: May lose important context/partial views
- **No padding handling**: Irregular crops get uniformly cropped

### **3. Limited Feature Utilization**
- Only RGB pixels from masked region
- **Ignored**: 
  - Segmentation mask quality/score
  - Spatial context (size, position on conveyor)
  - Multi-scale features
  - Boundary/edge information

### **4. Model Architecture**
- Generic ResNet50 without task-specific modifications
- No fine-tuning hyperparameters visible
- Single model - no ensemble or uncertainty estimation

### **5. Training Data Quality** (inferred)
- No data augmentation evident in inference
- Hard to assess if training data matches deployment conditions

---

## **Questions for Original Developer**

### **Training & Data Questions**
1. **Dataset composition**: How many clean vs pit examples? What's the class balance?
2. **Data collection conditions**: What lighting, camera settings, cherry varieties were used?
3. **Data splits**: Train/val/test split ratios? Any temporal or spatial splits?
4. **Augmentation**: What augmentations were used during training?
5. **Label quality**: Manual or automated? Inter-annotator agreement?

### **Model Architecture Questions**
6. **Why ResNet50 specifically?** Were other architectures tested (EfficientNet, MobileNet, custom CNN)?
7. **Pre-training**: Used ImageNet pretrained weights or trained from scratch?
8. **Input size**: Why 128×128? Was this optimized through experimentation?
9. **Mask processing**: Why dilate-erode? Was morphology tuned?

### **Training Process Questions**
10. **Loss function**: Binary cross-entropy with logits? Class weighting?
11. **Optimizer**: Adam? SGD? What learning rate schedule?
12. **Training duration**: How many epochs? Convergence behavior?
13. **Hyperparameters**: Batch size, learning rate, weight decay, dropout?
14. **Regularization**: Any techniques used to prevent overfitting?

### **Performance & Threshold Questions**
15. **Threshold selection**: How were 75%/50% thresholds determined? ROC analysis?
16. **Metrics achieved**: What accuracy, precision, recall, F1 on validation set?
17. **Confusion matrix**: What are the common error types (clean→pit vs pit→clean)?
18. **False positive/negative tolerance**: What's the business cost of each error type?
19. **Latency requirements**: What's the acceptable inference time per frame?

### **Operational Questions**
20. **Deployment drift**: How does performance change with different lighting or cherry batches?
21. **Failure modes**: What types of cherries does it consistently misclassify?
22. **Maintenance**: How often is retraining needed? Any model decay observed?
23. **Calibration**: Are softmax probabilities well-calibrated?

### **Improvement Questions**
24. **Known limitations**: What did you identify as the biggest weaknesses?
25. **Unexplored ideas**: What improvements did you consider but not implement?
26. **Resource constraints**: Any compute/memory limitations for model changes?

---

## **Suggested Improvement Approaches** (for discussion)

### **Quick Wins (Low Effort)**
- **Threshold calibration**: Optimize thresholds based on cost of errors
- **Probability calibration**: Temperature scaling or Platt scaling
- **Ensemble**: Use multiple thresholds or models for uncertainty estimation

### **Medium Effort**
- **Better preprocessing**: Adaptive cropping, smart padding, multi-scale inputs
- **Feature engineering**: Include segmentation confidence, size, position as features
- **Architecture tuning**: Try EfficientNet-B0, MobileNetV3 for speed/accuracy trade-offs

### **High Effort**
- **End-to-end training**: Train segmentation+classification jointly
- **Domain-specific backbone**: Custom architecture for cherry texture analysis
- **Multi-task learning**: Predict additional attributes (size, quality, variety)
