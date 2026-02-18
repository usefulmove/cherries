---
name: threading_ws Architecture Index
layer: System
impact_area: All
---

# threading_ws Architecture Index

**Status:** Production System (Current)  
**Location:** `/home/dedmonds/repos/traina/threading_ws/`  
**ROS2 Workspace:** Complete, self-contained  
**Previous System:** `cherry_system/` (Removed - see Migration Guide)

## Purpose

Production cherry processing system with HDR multi-layer imaging, 3-model AI pipeline (including stem detection), and C++-based orchestration.

## System Overview

This is a **complete, self-consistent ROS2 workspace** containing 16 packages that work together to:
1. Capture HDR multi-layer images from multiple cameras
2. Detect and classify cherries using 3-model AI (segmentation + classification + stem detection)
3. Track cherries on a moving conveyor belt
4. Coordinate robot picking operations
5. Monitor system health and performance

## Architecture Layers

| Layer | Packages | Responsibility |
|:------|:---------|:---------------|
| **Interface** | cherry_interfaces | Message/service definitions (16 msgs, 14 srvs, 3 actions) |
| **Acquisition** | cameras, trigger_node, plc_eip | Hardware interface (cameras, encoders, I/O) |
| **Processing** | cherry_detection, image_combiner, composite | AI detection and image processing |
| **Coordination** | composite, cherry_buffer | Orchestration and data buffering |
| **Execution** | fanuc_comms, tracking_projector | Robot communication and projection |
| **Utilities** | record_hdr, system_monitor, cherry_gui | Recording, monitoring, GUI |
| **External** | action_tutorials_cpp | ROS2 tutorials (not cherry-related) |

## Package Inventory

### Core Detection Pipeline

| Package | Type | Key Files | Purpose |
|:--------|:-----|:----------|:--------|
| **cherry_detection** | Python | ai_detector3.py, detector_node.py | 3-model AI detection (seg + clf + stem) |
| **cherry_interfaces** | Interface | msg/, srv/, action/ | HDR-capable interfaces |
| **composite** | C++ | composite_node.cpp | Main orchestration (action server) |
| **cherry_buffer** | Python | buffer_node.py | Cherry data buffering for robot sync |

### Hardware Interface

| Package | Type | Key Files | Purpose |
|:--------|:-----|:----------|:--------|
| **cameras** | C++ | camera_node.cpp, VimbaX SDK | Cognex camera driver |
| **trigger_node** | Python | trigger_node.py | Camera trigger generation |
| **plc_eip** | C++ | plc_node.cpp | PLC Ethernet/IP (encoder, I/O) |
| **fanuc_comms** | Python | fanuc_comms.py | Robot communication |

### Image Processing

| Package | Type | Key Files | Purpose |
|:--------|:-----|:----------|:--------|
| **image_combiner** | Python | combiner_node.py | Layer alignment and combination |
| **image_service** | Python | image_server.py | Image serving and caching |

### Tracking & Visualization

| Package | Type | Key Files | Purpose |
|:--------|:-----|:----------|:--------|
| **tracking_projector** | C++ | projector_node.cpp | Belt projection system |

### Utilities

| Package | Type | Key Files | Purpose |
|:--------|:-----|:----------|:--------|
| **record_hdr** | Python | record_node.py | HDR image recording |
| **system_monitor** | Python | monitor_node.py | Temperature monitoring |
| **cherry_gui** | C++ | (incomplete) | GUI application |
| **action_tutorials_cpp** | C++ | tutorial code | ROS2 examples |

## Key Capabilities

### HDR Multi-Layer Imaging
- **ImageSetHdr.msg**: Multi-layer image set with metadata
- **ImageLayer.msg**: Individual layer with encoder position
- Automatic layer alignment based on encoder counts

### 3-Model AI Pipeline
1. **Segmentation**: Mask R-CNN (`seg_model_red_v1.pt`) - Cherry detection
2. **Classification**: ResNet50 3-class (`classification-2_26_2025-iter5.pt`) - Quality assessment
3. **Stem Detection**: Faster R-CNN (`stem_model_10_5_2024.pt`) - Stem location

### Algorithm Switching
8 runtime-selectable algorithms:
- v1: fasterRCNN-Mask_ResNet50_V1
- v2: fasterRCNN-NoMask_ResNet50_6-12-2023
- v3: newlight-mask-12-15-2023
- v4: newlights-nomask-12-15-2023
- v5: NMS-nomask-1-3-2024
- **v6 (default): hdr_v1** ← Current production
- v7: hdr_v2
- v8: vote_v1

### Action-Based Orchestration
- **composite**: C++ action server (replaces control_node)
- Actions: Acquisitionhdr, FindCherries
- Async processing with feedback

## Model Files

Located in `threading_ws/src/cherry_detection/resource/` (excluded from git via .gitignore):

| Model | Size | Date | Algorithm | Purpose |
|:------|:-----|:-----|:----------|:--------|
| seg_model_red_v1.pt | 169M | 2024 | v6+ | Segmentation (Mask R-CNN) |
| classification-2_26_2025-iter5.pt | 91M | Feb 2025 | v6+ | Classification 3-class |
| stem_model_10_5_2024.pt | 166M | Oct 2024 | v6+ | Stem detection |
| ... (13 additional legacy models) | | | v1-v5 | Legacy algorithm support |

**Total:** ~1.1GB of model weights

## Discovery Protocol

Follow the enso discovery pattern:

1. **Start here** (this INDEX.md)
2. **Drill down** to specific package ARCHITECTURE.md files
3. **Follow code links** to implementation

### Package Documentation

| Package | Architecture Doc | Code Location |
|:--------|:----------------|:--------------|
| cherry_detection | [ARCHITECTURE.md](./cherry_detection/ARCHITECTURE.md) | `threading_ws/src/cherry_detection/` |
| cherry_interfaces | [ARCHITECTURE.md](./cherry_interfaces/ARCHITECTURE.md) | `threading_ws/src/cherry_interfaces/` |
| composite | [ARCHITECTURE.md](./composite/ARCHITECTURE.md) | `threading_ws/src/composite/` |
| cameras | [ARCHITECTURE.md](./cameras/ARCHITECTURE.md) | `threading_ws/src/cameras/` |
| cherry_buffer | [ARCHITECTURE.md](./cherry_buffer/ARCHITECTURE.md) | `threading_ws/src/cherry_buffer/` |
| fanuc_comms | [ARCHITECTURE.md](./fanuc_comms/ARCHITECTURE.md) | `threading_ws/src/fanuc_comms/` |
| plc_eip | [ARCHITECTURE.md](./plc_eip/ARCHITECTURE.md) | `threading_ws/src/plc_eip/` |
| trigger_node | [ARCHITECTURE.md](./trigger_node/ARCHITECTURE.md) | `threading_ws/src/trigger_node/` |

## System Entry Points

### For Detection Pipeline
→ Start at [cherry_detection/ARCHITECTURE.md](./cherry_detection/ARCHITECTURE.md)

### For Message Types
→ Start at [cherry_interfaces/ARCHITECTURE.md](./cherry_interfaces/ARCHITECTURE.md)

### For Orchestration Flow
→ Start at [composite/ARCHITECTURE.md](./composite/ARCHITECTURE.md)

## Migration Context

This system **replaces** the legacy `cherry_system/`:

| Aspect | cherry_system (Legacy) | threading_ws (Production) |
|:-------|:-----------------------|:--------------------------|
| Detection | 2-model | 3-model (+stem) |
| Classification | 2-class | 3-class (clean/maybe/pit) |
| Interface | Simple | HDR multi-layer |
| Orchestration | control_node (Python) | composite (C++) |
| Camera | AVT Vimba | Cognex |
| I/O | usb_io | plc_eip |

**Full migration details:** [MIGRATION_cherry_system_to_threading_ws.md](../../reference/MIGRATION_cherry_system_to_threading_ws.md)

## Build Instructions

```bash
cd /home/dedmonds/repos/traina/threading_ws
colcon build
source install/setup.bash
```

## Related Documentation

- [Migration Guide](../../reference/MIGRATION_cherry_system_to_threading_ws.md)
- [Legacy System (cherry_system)](../cherry_system/ARCHITECTURE.md)
- [Training Data](../../reference/training-data.md)
- [Stem Detection](../../core/architecture/inference_pipeline/STEM_DETECTION.md)

## Maintenance Notes

- **Model files** are excluded from git (.gitignore) - keep backup copy
- **cherry_gui** is incomplete (no package.xml)
- **action_tutorials_cpp** is ROS2 tutorial code (not cherry-related)
- All other packages are production-ready

## Questions?

See [Open Questions: Stem Detection](../../reference/open-questions-stem-detection.md) for active investigations.
