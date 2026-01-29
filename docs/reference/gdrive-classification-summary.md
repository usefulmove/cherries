## ResNet-50 Training Documentation on Google Drive Shares

### 1. **ResNet-50 Implementation Code**
- **Location**: `/home/dedmonds/base/gdrive/Cherry Line Files/cherry_backup_12_21_2022/src/cherry_system/cherry_detection/cherry_detection/ai_detector.py`
- **Key Details**:
  - Line 148: `self.classifier = resnet50().to(self.device)`
  - Lines 149-152: Custom classifier head with 2 classes (clean cherry vs pit)
  - Pre-trained ResNet-50 backbone with fine-tuned final layer

### 2. **Training Infrastructure**
- **Engine File**: `/home/dedmonds/base/gdrive/Cherry Line Files/cherry_backup_12_21_2022/install/cherry_detection/lib/python3.10/site-packages/cherry_detection/engine.py`
  - Contains `train_one_epoch()` function for training
  - Includes evaluation functions with COCO API integration
  - Supports warmup learning rate scheduling

### 3. **Training Data Organization**
- **Location**: `/home/dedmonds/base/gdrive/Traina Cherry Pit Project/`
- **Dataset Structure**:
  - `cherry_clean/` - 800+ labeled training images of clean cherries
  - `cherry_pit/` - Labeled training images of cherry pits  
  - `training_datasets/` - (Currently empty)
  - Multiple date-based data folders: `11_2_data/`, `12_6/`, `20230104/`, etc.

### 4. **Training Configuration**
- **Model Parameters**: 
  - 2-class classification (clean vs pit)
  - Custom image preprocessing (128x128 CenterCrop)
  - GPU acceleration support
  - Softmax probability thresholding for classification confidence

### 5. **Evaluation/Validation**
- **Jupyter Notebooks**: 
  - `Copy of cherry_classification_evaluation.ipynb` in both folders
  - Multiple model variants mentioned (gradient descent vs ADAM optimizer)

### Summary
The ResNet-50 fine-tuning setup uses a standard transfer learning approach with:
- Pre-trained ResNet-50 backbone
- Custom 2-class classification head
- Organized training dataset with labeled images
- Training engine with epoch-based training loop
- Evaluation framework for model comparison
