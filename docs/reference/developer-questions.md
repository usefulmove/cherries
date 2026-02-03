# Developer Questions for Meeting

## Priority Questions (Based on Optimization Experiments)

### 1. Training Data & Class Distribution
- **What is the distribution of clean vs pit in your training data?**
  - Our analysis shows 54% clean, 46% pit (well-balanced)
  - Has this ratio changed in production data?
  - Are there specific orientations/lighting conditions causing errors?

### 2. Fine-Tuning Approach
- **Have you tried fine-tuning beyond just the FC (fully connected) layer?**
  - Currently only the final classifier is trained while all other layers stay frozen with ImageNet weights
  - Unfreezing layer3-4 (last residual blocks) with differential learning rates could capture better pit-specific features
  - We're testing: layer4=1e-5, fc=1e-3 (see optimization experiments notebook)

### 3. Input Resolution & Detail
- **Would 224×224 crops give better pit detection detail?**
  - Current: 128×128 pixels
  - Trade-off: More computation but better feature extraction from pretrained weights
  - Could reveal finer pit textures that 128×128 might miss

### 4. Decision Thresholds
- **Why the "maybe" category (probability 0.5-0.75)?**
  - Current thresholds: pit≥0.75, maybe≥0.5, clean≥0.5
  - Could we eliminate "maybe" or adjust based on precision-recall analysis?
  - We're testing optimal thresholds for ≥95% pit recall (minimize missed pits)

### 5. Cherry Characteristics Causing Errors
- **Are there specific cherry orientations or lighting conditions causing misclassification?**
  - This would guide targeted data augmentation strategies
  - Current augmentation: rotation, affine, color jitter
  - What edge cases should we focus on?

## Historical Context Questions

### 6. Original Training Setup
- **How was training and testing the model originally done?**
  - Reference to "Copy of cherry_classification_evaluation.ipynb" document?
  - What was the training script and hyperparameters used?
  - Adam vs SGD optimizer - what did you try?

### 7. Training Data Timeline
- **Do you have access to more recent training images?**
  - When were the current training images captured?
  - Has cherry variety, camera setup, or lighting changed since then?
  - Could collecting new production data improve accuracy?

### 8. Unexplored Improvements
- **What improvements did you consider but not implement?**
  - Alternative architectures (EfficientNet, Vision Transformers)?
  - Ensemble methods?
  - Cherry size as an additional feature?
  - Multi-stage classification with confidence-based routing?

### 9. System Constraints
- **Compute or memory limitations? Speed/accuracy trade-offs?**
  - Current: 16.7ms CPU latency (ResNet50)
  - Is faster inference needed for higher throughput?
  - Would you trade 1-2% accuracy for 40% faster inference (ResNet18)?

### 10. Production Deployment
- **Current production model details:**
  - Where are the production fine-tuned model parameter files?
  - What's the actual production accuracy being measured?
  - How are false negatives (missed pits) handled downstream?

## Setup & Integration

### 11. System Setup
- **Instructions for setting up and running the system from scratch?**
  - Hardware configuration (cameras, pneumatics, compute)
  - Software dependencies and versions
  - Calibration procedures

### 12. Code Repository
- **Is there a central code repository?**
  - Version control for models and training scripts?
  - How are model updates deployed to production?

### 13. Additional Context
- **Is there anything else that is important to know?**
  - Upcoming changes to hardware or cherry varieties?
  - Business metrics for success (throughput, yield, safety)?
  - Timeline for deploying improvements?

---

## Notes from Optimization Experiments

We've prepared three experiments to explore improvements:

1. **Differential Learning Rates** - Fine-tune deeper layers with layer-specific LRs
2. **Threshold Optimization** - Find optimal decision boundaries for ≥95% pit recall
3. **ResNet18 Backbone** - Test smaller/faster model (11.7M vs 25.6M params)

All experiments use:
- Augmentation (rotation, affine, color jitter)
- Unnormalized training (0-255 to match production)
- Same 128×128 input size

**Notebook:** `training/notebooks/colab_optimization_experiments.ipynb`
