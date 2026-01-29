# SanDisk Drive Contents

## **Core Code Repositories**

**`threading_ws/`** (PRIMARY - started April 2024)
- This is your **active production ROS2 workspace**
- Key subdirectories:
  - `src/cherry_detection/` - Detection/classification models & inference
  - `src/control_node/` - Main control logic
  - `src/image_pipeline/` - Camera handling & image processing
  - `src/ros2_performance/` - Performance monitoring
- Contains: 74 Python files, launch files, config files

**`cherry_ws/`** (OLDER - monolithic version)
- `src/cherry_detection/` - Similar to threading_ws but older architecture
- Legacy code - not actively maintained

---

## **Training & ML Infrastructure**

**`pytorch/`** - Training infrastructure
- `vision/` - PyTorch Vision library (models, training scripts)
- `yolov7/` - YOLOv7 detection training
- `cherry_data/` - 566 annotated images (YOLOv7 format)
- 27 Jupyter notebooks for training/experimentation

---

## **Training Data - CLASSIFICATION**

**`/Pictures/hdr/`** (11.8 GB, 11,800+ images)
- **This is your raw training data** for classification
- Organized by cherry type and pit status:
  ```
  natural_pits/ (258 images)
  natural_clean/ (102 images)
  organic_pits/ (189 images)
  organic_clean/ (127 images)
  sulfur_pits/
  sulfur_clean/
  20240611_clean/ (1,700+ images - recent collection)
  20240423 missed pits/ (error cases)
  20240923 stems/ (edge cases)
  ```
- **NOT auto-labeled** - manually collected and categorized

**`/Pictures/others/`** - Additional raw images

---

## **Training Data - DETECTION**

**`pytorch/cherry_data/`** (1.2 GB)
- COCO/VOC/YOLOv7 format annotations
- 394 train + 113 val + 59 test images
- Classes: null_val, with_pit, clean

**Multiple Roboflow exports** in the drive:
- `Cherry Inspection.v3i.*` (various formats)
- `Cherry inspection -2.v3i.*`

---

## **Model Weights - PRODUCTION**

**In `threading_ws/src/cherry_detection/resource/`** (Active production models):
```
cherry_classification.pt              # ResNet50 baseline
classification-202406031958-resnet50-adam-2.pt  # Recent ResNet50
classification-2_26_2025-iter5.pt     # Most recent (Feb 26 2025)
classification-iter10.pt              # Earlier version
```

**Legacy models scattered across drive**:
- `Desktop/`, `Documents/`, `Documents/simplify/`
- Various experiments (exp1-exp22 in yolov7 runs)

---

## **Documentation on Drive**

**Scattered throughout**:
- `Desktop/auto_label_semantic.ipynb` - LabelMe workflow
- `Desktop/how to start up program` - Startup instructions
- `Desktop/image save read me.txt` - Image collection docs
- `Desktop/change_algorithm.txt` - Algorithm switching
- `Documents/` - Various technical notes and backups
- READMEs in `image_pipeline/`, `yolov7/`, dataset directories

---

## **Configuration Files**

- **Camera configs**: Mako_G-319.yaml, Mako_G-507.yaml, USB configs
- **Hardware calibration**: Origin (2448, 652, π/2), Scaling 2710.3 px/m
- **Algorithm parameters**: Dynamic ROS2 parameters (maybe_threshold=0.04, pick_threshold=0.06)
