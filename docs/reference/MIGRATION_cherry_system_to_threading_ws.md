# Cherry Processing System: Migration Guide

**cherry_system (Legacy)** → **threading_ws (Production)**

**Date:** 2025-02-05  
**Status:** threading_ws copied and operational, cherry_system archived for reference

---

## Executive Summary

The production cherry processing system has migrated from `cherry_system/` to `threading_ws/`. This represents a **major architectural upgrade** with significant improvements in detection accuracy, processing pipeline, and hardware integration.

**Note on Large Files:** Rosbag test data (`.db3`) and VimbaX SDK (121MB) are excluded from git due to GitHub's 100MB limit. See [OPEN_ISSUES.md](./OPEN_ISSUES.md) for details.

### Key Changes at a Glance

| Aspect | Before (cherry_system) | After (threading_ws) | Impact |
|:-------|:----------------------|:---------------------|:-------|
| **AI Models** | 2-model (seg + clf) | **3-model (+ stem)** | Stem detection added |
| **Classification** | 2-class (clean/pit) | **3-class (+maybe)** | Explicit uncertainty handling |
| **Imaging** | Single image | **HDR multi-layer** | Better quality, alignment |
| **Orchestration** | control_node (Python) | **composite (C++)** | Async action-based |
| **Detection Interface** | Detection.srv | **Detectionhdr.srv** | Breaking change |
| **Camera Driver** | AVT Vimba | **Cognex** | Different hardware |
| **I/O System** | usb_io | **plc_eip** | PLC-based control |
| **Packages** | 8 packages | **16 packages** | Expanded capabilities |

---

## Package Comparison

### Added in threading_ws (8 new packages)

| Package | Purpose | Critical? | Location |
|:--------|:--------|:----------|:---------|
| **cherry_buffer** | Cherry data buffering for robot synchronization | **Yes** | `threading_ws/src/cherry_buffer/` |
| **composite** | Main orchestration node (C++ action server) | **Yes** | `threading_ws/src/composite/` |
| **plc_eip** | PLC Ethernet/IP communication (encoder, I/O) | **Yes** | `threading_ws/src/plc_eip/` |
| **cherry_gui** | GUI application (incomplete) | No | `threading_ws/src/cherry_gui/` |
| **image_combiner** | Multi-layer image alignment and combination | Yes | `threading_ws/src/image_combiner/` |
| **image_service** | Image serving and caching service | Yes | `threading_ws/src/image_service/` |
| **record_hdr** | HDR image recording utility | No | `threading_ws/src/record_hdr/` |
| **system_monitor** | Temperature and system health monitoring | No | `threading_ws/src/system_monitor/` |
| **trigger_node** | Camera trigger generation based on encoder | **Yes** | `threading_ws/src/trigger_node/` |
| **action_tutorials_cpp** | ROS2 tutorial examples | No | `threading_ws/src/action_tutorials_cpp/` |

### Modified Packages (4 significantly changed)

| Package | cherry_system | threading_ws | Changes |
|:--------|:--------------|:-------------|:--------|
| **cherry_detection** | 2-model, 2-class, simple detection | **3-model, 3-class, HDR, stem detection, 8 algorithms** | Complete rewrite |
| **cherry_interfaces** | 4 msgs, 6 srvs, 2 actions | **16 msgs, 14 srvs, 3 actions** | Major expansion |
| **fanuc_comms** | Simple Detection service client | **Multiple services, cherry buffer integration** | Enhanced |
| **cameras** | AVT Vimba camera driver | **Cognex camera driver** | Hardware change |

### Removed (Not in threading_ws)

| Package | Replacement | Notes |
|:--------|:------------|:------|
| **control_node** | **composite** | Different architecture - C++ vs Python |
| **camera_simulator** | **None** | Simulation capability removed |
| **avt_vimba_camera** | **cameras** | Different camera vendor |
| **usb_io** | **plc_eip** | Different I/O architecture |

---

## Interface Changes (Breaking)

### Messages: Expanded from 4 to 16

**Kept (4 messages - backward compatible):**
- Cherry.msg
- CherryArray.msg
- CherryArrayStamped.msg
- ImageSet.msg (legacy)

**Added (12 new messages):**
- ImageLayer.msg - Single layer in HDR set
- ImageSetHdr.msg - Multi-layer HDR image set with metadata
- Trigger.msg - Frame trigger with encoder data
- EncoderCount.msg - Encoder position
- HSC.msg - High-speed counter
- PickMode.msg - Robot pick mode configuration
- Inputs.msg - Digital input states
- Outputs.msg - Digital output states
- Temperature.msg - System temperature

### Services: Expanded from 6 to 14

**Kept (6 services):**
- Detection.srv (legacy - for backward compatibility)
- CombineImages.srv
- EncoderCount.srv
- ImageSave.srv
- SetLatch.srv
- SetLights.srv

**Added (8 new services):**
- Detectionhdr.srv - **Primary detection service for HDR**
- EncoderLatches.srv - Get latched encoder values
- GetCherryBuffer.srv - Get buffered cherries for robot
- LatchRobot.srv - Robot latch status
- ResetLatches.srv - Clear encoder latches
- ImageCounts.srv - Image counter query
- Trigger.srv - Manual trigger

### Actions: Expanded from 2 to 3

**Kept (2 actions):**
- Acquisition.action (legacy)

**Added (1 new action):**
- Acquisitionhdr.action - HDR image acquisition
- FindCherries.action - **Complete detect-and-track action**

### Critical Breaking Changes

| Old Interface | New Interface | Migration Required |
|:--------------|:--------------|:-------------------|
| `Detection.srv` (single image) | `Detectionhdr.srv` (ImageSetHdr) | **Yes - all callers** |
| `Acquisition.action` | `Acquisitionhdr.action` | Yes - if used |
| `ImageSet.msg` | `ImageSetHdr.msg` | Yes - pipeline nodes |

---

## AI/ML Model Changes

### Detection Pipeline Evolution

**cherry_system (Legacy 2-Model):**
```
Single Image → Mask R-CNN (cherry_segmentation.pt) → Crops → ResNet50 (cherry_classification.pt)
                                                                  ↓
                                                            2-class (clean/pit)
                                                            Threshold-based maybe
```

**threading_ws (Production 3-Model):**
```
ImageSetHdr → Layer Alignment → Mask R-CNN (seg_model_red_v1.pt) → Crops → ResNet50 (classification-2_26_2025-iter5.pt)
     ↓                                                                  ↓
Faster R-CNN (stem_model_10_5_2024.pt)                          3-class (clean/maybe/pit)
     ↓                                                                  ↓
Stem Detection                                                    Explicit classes
```

### Model Files Comparison

| Model | cherry_system | threading_ws | Changes |
|:------|:--------------|:-------------|:--------|
| **Segmentation** | cherry_segmentation.pt (169M) | seg_model_red_v1.pt (169M) | Updated |
| **Classification** | cherry_classification.pt (90M) | classification-2_26_2025-iter5.pt (91M) | **3-class, Feb 2025** |
| **Stem Detection** | **(none)** | stem_model_10_5_2024.pt (166M) | **NEW** |

### Algorithm Versions

**cherry_system:** Single algorithm

**threading_ws:** 8 selectable algorithms via parameter
- v1: fasterRCNN-Mask_ResNet50_V1
- v2: fasterRCNN-NoMask_ResNet50_6-12-2023
- v3: newlight-mask-12-15-2023
- v4: newlights-nomask-12-15-2023
- v5: NMS-nomask-1-3-2024
- **v6 (default): hdr_v1** ← Current production
- v7: hdr_v2
- v8: vote_v1

### Classification Categories

**cherry_system (2-class with thresholds):**
| Label | Category | Threshold |
|:------|:---------|:----------|
| 1 | Clean | clean_prob ≥ 0.5 |
| 2 | Pit | pit_prob ≥ 0.75 |
| 3 | Side | Edge detection |
| 5 | Maybe | 0.5 ≤ pit_prob < 0.75 |

**threading_ws (3-class explicit):**
| Label | Category | Model Output | Description |
|:------|:---------|:-------------|:------------|
| 1 | Clean | Class 0 | No pit |
| 2 | Pit | Class 2 | Has pit |
| 3 | Side | Position | Edge of belt |
| 5 | Maybe | Class 1 | Uncertain |
| **6** | **Stem** | **Stem model** | **Stem detected** |

---

## Architecture Differences

### Orchestration: Sequential vs Async

**cherry_system (control_node - Sequential):**
```
control_node (Python)
    ↓
Call Detection service
    ↓
Wait for results
    ↓
Publish to fanuc_comms
```

**threading_ws (composite - Async Actions):**
```
composite (C++ action server)
    ↓
Start FindCherries.action
    ↓
Parallel: trigger_node → cameras → image_combiner
    ↓
cherry_detection (Detectionhdr)
    ↓
cherry_buffer (caching)
    ↓
fanuc_comms (picks from buffer)
```

### Data Flow: Simple vs HDR

**cherry_system:**
- Single camera image
- Direct detection call
- Immediate results

**threading_ws:**
- Multi-layer HDR (top2, bot1, bot2)
- Encoder-based alignment
- Layer combination
- Detection with metadata
- Buffered results for robot sync

---

## Code Examples: Migration

### Calling Detection (Old → New)

**cherry_system:**
```python
from cherry_interfaces.srv import Detection

client = self.create_client(Detection, '/detect')
request = Detection.Request()
request.image = image_msg
response = await client.call_async(request)
cherries = response.cherries
```

**threading_ws:**
```python
from cherry_interfaces.srv import Detectionhdr
from cherry_interfaces.msg import ImageSetHdr, ImageLayer, Trigger

client = self.create_client(Detectionhdr, '/cherry_detection/detect')
request = Detectionhdr.Request()
request.frame_id = 12345
request.image_top2 = top_image
request.image_bot1 = bot1_image
request.image_bot2 = bot2_image
request.count_bot1 = encoder_count1
request.count_bot2 = encoder_count2
request.mm_bot1 = mm1
request.mm_bot2 = mm2
# ... set other fields
response = await client.call_async(request)
cherries = response.cherries
```

### Publishing Results (Old → New)

**cherry_system:**
```python
from cherry_interfaces.msg import CherryArrayStamped

msg = CherryArrayStamped()
msg.header = header
msg.cherries = cherries
publisher.publish(msg)
```

**threading_ws:**
```python
from cherry_interfaces.msg import CherryArray

msg = CherryArray()
msg.cherries = cherries  # Can include Type 6 (stem)
publisher.publish(msg)
```

---

## File Paths Reference

### Legacy System (Archived)
```
/home/dedmonds/repos/traina/cherry_system/
├── cherry_detection/          # Old 2-model detection
├── cherry_interfaces/         # Basic interfaces (4/6/2)
├── control_node/              # Python orchestration
├── fanuc_comms/               # Robot communication
├── tracking_projector/        # Projection (unchanged)
├── camera_simulator/          # Simulation (removed)
├── avt_vimba_camera/          # AVT driver (removed)
└── usb_io/                    # USB I/O (removed)
```

### Production System (Current)
```
/home/dedmonds/repos/traina/threading_ws/src/
├── cherry_detection/          # 3-model with stem, 8 algorithms
├── cherry_interfaces/         # Expanded (16/14/3)
├── cherry_buffer/             # NEW: Data buffering
├── composite/                 # NEW: C++ orchestration
├── plc_eip/                   # NEW: PLC communication
├── fanuc_comms/               # Updated for HDR
├── tracking_projector/        # Unchanged
├── cameras/                   # Cognex driver
├── trigger_node/              # NEW: Trigger generation
├── image_combiner/            # NEW: Layer combination
├── image_service/             # NEW: Image serving
├── record_hdr/                # NEW: Recording
├── system_monitor/            # NEW: Monitoring
└── action_tutorials_cpp/      # Tutorial code
```

---

## Build Instructions

### threading_ws (Production)
```bash
cd /home/dedmonds/repos/traina/threading_ws
colcon build
source install/setup.bash
```

### Model Files
All .pt files are in `threading_ws/src/cherry_detection/resource/` (~1.1GB total, excluded from git).

---

## Migration Checklist

### For Package Developers

- [ ] Update service calls: Detection → Detectionhdr
- [ ] Update message types: ImageSet → ImageSetHdr where needed
- [ ] Add layer alignment if processing multi-layer images
- [ ] Handle Type 6 (stem) in cherry processing logic
- [ ] Test with recorded HDR rosbags

### For System Operators

- [ ] Verify PLC communication (replaces USB I/O)
- [ ] Test camera triggers with trigger_node
- [ ] Validate encoder counting
- [ ] Check cherry_buffer for robot sync
- [ ] Monitor system with system_monitor

### For Model Training

- [ ] Use 3-class labels: 0=clean, 1=maybe, 2=pit
- [ ] Train on unnormalized data (0-255)
- [ ] Collect stem training data if needed
- [ ] Validate models on HDR images

---

## Open Questions

See [open-questions-stem-detection.md](../reference/open-questions-stem-detection.md) for active investigations:
- Practical use of stem detection in sorting
- Robot integration for Type 6 (stem) messages
- Performance impact of 3-model vs 2-model

---

## Related Documentation

- [threading_ws Architecture](../docs/core/architecture/threading_ws/INDEX.md)
- [cherry_interfaces Reference](../docs/core/architecture/threading_ws/cherry_interfaces/ARCHITECTURE.md)
- [Stem Detection](../docs/core/architecture/inference_pipeline/STEM_DETECTION.md)
- [Training Data](../docs/reference/training-data.md)

---

## Summary

The migration from `cherry_system` to `threading_ws` represents a **complete system upgrade** with:
- **Better detection:** 3-model AI with stem detection
- **Better imaging:** HDR multi-layer processing
- **Better orchestration:** Async action-based control
- **Better integration:** PLC-based I/O, buffered robot communication

**Both systems coexist** in the repository. Use `threading_ws` for production, reference `cherry_system` for legacy compatibility.
