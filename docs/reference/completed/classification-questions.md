# Classification Questions

## Model Architecture & Configuration
1. What's the current ResNet50 configuration, and was it fine-tuned or trained from scratch?
2. Are the current 128×128 input crops sufficient, or would 224×224 provide better detail for pit detection?
3. Was ImageNet normalization applied during training (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])?
4. Which layers of ResNet50 were unfrozen during training (just FC layer, or earlier blocks too)?
5. Are there class imbalance issues between clean and pit samples?

## Data Pipeline
6. What's the size and quality of the training dataset?
7. How were the training images annotated (manual labeling, automated collection)?
8. Was data augmentation used during training (rotation, flips, color jitter)?
9. Are there existing validation metrics or confusion matrices available?
10. How does the Mask R-CNN segmentation quality affect classification accuracy?

## Inference Pipeline
11. Does the current inference code apply the same preprocessing as training?
12. Why is there a "maybe" category (label 5) between clean (≥50%) and pit (≥75%)?
13. Are the current decision thresholds (0.5/0.75) optimal for the production environment?
14. How do edge detections (label 3) impact overall system performance?

## Training Infrastructure
15. Where are the original training scripts or notebooks located?
16. What hyperparameters were used (learning rate, batch size, epochs)?
17. Is there version control for model weights beyond `cherry_classification.pt`?
18. What was the validation accuracy/precision/recall during training?

## Evaluation & Metrics
19. What's the current production accuracy, and how is it measured?
20. What types of misclassifications are most common (false positives vs false negatives)?
21. Are there specific cherry orientations/lighting conditions that cause errors?
22. How does the system handle ambiguous cases currently?

## Improvement Strategy
23. Which of the identified improvements would yield the best ROI?
24. What are the trade-offs between accuracy improvements and inference speed?
25. Is there access to labeled test data for validation?
26. What's the business impact of misclassifications (missed pits vs false pits)?

## System Integration
27. How does classification accuracy affect downstream actuation decisions?
28. Can the system tolerate a "reinspect" or "uncertain" category?
29. What's the inference latency budget per frame?
30. Are there hardware constraints (GPU memory, CPU) for model size?

The `resnet50-analysis.md` document in `/docs/` provides a detailed prioritized list of potential improvements you can reference.
