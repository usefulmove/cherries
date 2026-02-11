# System Architecture Quick Reference

**Cherry Processing System - Real-time Sorting Pipeline**

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHERRY PROCESSING SYSTEM                              │
│                     ROS2 Humble • Real-time Sorting                         │
└─────────────────────────────────────────────────────────────────────────────┘

PHYSICAL WORLD                        SOFTWARE PIPELINE
     │                                       │
     │ Cherry on Belt                       │
     ▼                                       ▼
┌─────────────┐    Encoder Ticks      ┌──────────────────┐
│  ENCODER    │──────────────────────▶│  usb_io_node     │
│  (Hardware) │   /encoder_ticks      │  (Hardware I/O)  │
└─────────────┘                       └────────┬─────────┘
                                               │
                                               │ Service Call
                                               ▼
                                       ┌──────────────────┐
                                       │  control_node    │
                                       │  (The Brain)     │
                                       │                  │
                                       │  • Tracks belt   │
                                       │    position      │
                                       │  • Triggers      │
                                       │    cameras       │
                                       │  • Manages       │
                                       │    cherry queue  │
                                       └────────┬─────────┘
                                                │
                      ┌─────────────────────────┼──────────────────────────┐
                      │                         │                          │
                      ▼                         ▼                          ▼
              ┌───────────────┐       ┌──────────────────┐      ┌──────────────────┐
              │ TOP CAMERA    │       │ avt_vimba_camera │      │ BOTTOM CAMERA   │
              │ (Strobe Sync) │◀──────│  (Vision Layer)  │─────▶│ (Strobe Sync)    │
              └───────┬───────┘       └──────────────────┘      └────────┬─────────┘
                      │                         │                          │
                      │    /top_image           │                   /bottom_image
                      │                         ▼                          │
                      │                 ┌──────────────────┐               │
                      │                 │  Image Assembly  │               │
                      │                 │  (3-Channel      │               │
                      │                 │   Composite)     │               │
                      │                 │                  │               │
                      │                 │  R = Top Image   │               │
                      │                 │  G = Black       │               │
                      │                 │  B = Bottom      │               │
                      │                 │     Image        │               │
                      │                 └────────┬─────────┘               │
                      │                          │                         │
                      └──────────────────────────┼─────────────────────────┘
                                                 │
                                                 │ Service: DetectCherries
                                                 ▼
                                       ┌──────────────────┐
                                       │ cherry_detection │
                                       │ (Inference)      │
                                       │                  │
                                       │  ┌──────────────┴──┐
                                       │  │ Stage 1:        │
                                       │  │ Mask R-CNN      │
                                       │  │ (Segmentation)  │
                                       │  │ • ResNet50-FPN  │
                                       │  │ • 2 classes     │
                                       │  └────────┬────────┘
                                       │           │
                                       │           ▼
                                       │  ┌──────────────────┐
                                       │  │ Stage 2:         │
                                       │  │ ResNet50         │
                                       │  │ (Classification) │
                                       │  │ • 128×128 crops  │
                                       │  │ • Clean/Pit/Maybe│
                                       │  └────────┬─────────┘
                                       │           │
                                       │           ▼
                                       │  ┌──────────────────┐
                                       │  │ Labels:          │
                                       │  │ 1 = Clean        │
                                       │  │ 2 = Pit (≥0.75)  │
                                       │  │ 3 = Edge         │
                                       │  │ 5 = Maybe (0.5)  │
                                       │  └──────────────────┘
                                       └────────┬─────────┘
                                                │
                                                │ Service Response
                                                │ (classifications)
                                                ▼
                                       ┌──────────────────┐
                                       │  control_node    │
                                       │  (Decision)      │
                                       │                  │
                                       │  • Maps coords   │
                                       │    to belt       │
                                       │  • Decides       │
                                       │    sort/reject   │
                                       └────────┬─────────┘
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                         ▼                      ▼                      ▼
               ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
               │  fanuc_comms     │   │   usb_io_node    │   │tracking_projector│
               │  (Robot Control) │   │  (Air Jet)       │   │  (Visualization) │
               │                  │   │                  │   │                  │
               │  TCP/IP Socket   │   │  Pneumatic       │   │  OpenGL/Projector│
               │  to Fanuc        │   │  Ejector         │   │  • Green=Clean   │
               │  Robot           │   │                  │   │  • Red=Pit       │
               │                  │   │                  │   │  • Tracks moving │
               │  (x,y,class)     │   │  Trigger at      │   │    cherries      │
               │                  │   │  belt position   │   │                  │
               └──────────────────┘   └──────────────────┘   └──────────────────┘
                         │                      │                      │
                         ▼                      ▼                      ▼
               ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
               │  ROBOT ARM       │   │  AIR JET         │   │  PROJECTOR       │
               │  (Sort Good)     │   │  (Reject Bad)    │   │  (Visual Feedback)│
               └──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## Timing & Latency Budget

| Stage | Budget | Actual | Notes |
|-------|--------|--------|-------|
| **Total Pipeline** | < Belt Travel Time | ~50-100ms | Must complete before ejector |
| **Vision Capture** | < 5ms | ~2-3ms | Strobe + camera readout |
| **Image Assembly** | < 2ms | ~1ms | Composite creation |
| **Segmentation (Mask R-CNN)** | < 15ms | ~8-12ms | GPU inference |
| **Classification (ResNet50)** | < 16ms | ~16ms (CPU baseline) | GPU inference (faster) |
| **Tracking/Decision** | < 5ms | ~1-2ms | Coordinate mapping |
| **Robot/Actuation** | < 50ms | ~20-30ms | Network + mechanical |

**Key Constraint:** Camera-to-Ejector belt travel time determines total budget.

---

## Node Directory

| Node | Package | Type | Key Topics/Services | Responsibility |
|------|---------|------|---------------------|----------------|
| **usb_io_node** | hardware_io | ROS2 Node | Sub: encoder_ticks<br>Srv: io_control | Hardware interface for encoders, strobes, pneumatics |
| **control_node** | control_node | ROS2 Node | Srv: DetectCherries<br>Pub: sort_commands, tracking_state | Central orchestrator - the "brain" |
| **avt_vimba_camera** | avt_vimba | ROS2 Node | Pub: /top_image, /bottom_image | Camera driver ( Allied Vision ) |
| **cherry_detection** | cherry_detection | ROS2 Node | Srv: DetectCherries | ML inference service |
| **fanuc_comms** | fanuc_comms | ROS2 Node | TCP socket to robot | Robot controller interface |
| **tracking_projector** | tracking_projector | C++/Qt Node | Sub: tracking_state | Real-time belt visualization |

---

## Key File Locations

### Models (CRITICAL - Verify These!)

| Model | Canonical Path | Size | Status |
|-------|---------------|------|--------|
| **Classification (Production)** | `cherry_system/cherry_detection/resource/cherry_classification.pt` | ~90MB | ⚠️ VERIFY ACTIVE |
| **Classification (Duplicate)** | `cherry_system/control_node/resource/cherry_classification.pt` | ~90MB | ⚠️ BUG: May load instead |
| **Classification (Best Training)** | `training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt` | ~90MB | 94.05% accuracy |
| **Classification (ResNet18)** | `training/experiments/resnet18_augmented_unnormalized/model_best.pt` | ~43MB | 91.92% accuracy |
| **Segmentation** | `cherry_system/cherry_detection/resource/cherry_segmentation.pt` | ~168MB | Mask R-CNN |

**⚠️ Known Bug:** Model may load from `control_node/resource` instead of `cherry_detection/resource`. Always verify at runtime.

### Code

| Component | Path | Key Files |
|-----------|------|-----------|
| **Classification** | `cherry_system/cherry_detection/` | ai_detector.py, detector_node.py |
| **Control Logic** | `cherry_system/control_node/` | control_node.py |
| **Training** | `training/` | scripts/train.py, notebooks/ |
| **Interfaces** | `cherry_system/cherry_interfaces/` | Custom ROS2 messages/services |

---

## Message Flow Summary

```
1. Encoder tick → control_node (belt position tracking)
2. control_node → avt_vimba_camera (trigger capture)
3. Camera → control_node (top + bottom images)
4. control_node → cherry_detection (DetectCherries service)
5. cherry_detection → control_node (classifications + coords)
6. control_node → fanuc_comms (sort command)
7. control_node → usb_io_node (ejector trigger)
8. control_node → tracking_projector (visualization)
```

---

## Critical System Parameters

| Parameter | Current Value | Impact |
|-----------|--------------|--------|
| **Classification Thresholds** | pit≥0.75, maybe≥0.5, clean≥0.5 | Decision boundaries |
| **Mask R-CNN Score Threshold** | 0.5 | Detection sensitivity |
| **Mask R-CNN NMS Threshold** | 0.5 | Overlap suppression |
| **Input Crop Size** | 128×128 | Classification input |
| **Inference Device** | NVIDIA GPU (model TBD) | CUDA-enabled PyTorch |
| **Belt Speed** | [TBD] | Determines latency budget |

**Latency Note:** Baseline benchmarks (16ms) were measured on CPU for development/comparison purposes. Production system uses NVIDIA GPU for inference. GPU latency benchmarks pending hardware confirmation.

---

## Architecture Layers

| Layer | Components | Responsibility |
|-------|-----------|--------------|
| **Hardware I/O** | usb_io_node | Physical interfacing (encoders, strobes, pneumatics) |
| **Vision Acquisition** | avt_vimba_camera | Camera drivers, image capture, synchronization |
| **Tracking & Orchestration** | control_node | Belt tracking, timing, decision logic |
| **Inference Pipeline** | cherry_detection | Segmentation (Mask R-CNN) + Classification (ResNet50) |
| **Actuation** | fanuc_comms, usb_io_node | Robot sorting, air jet rejection |
| **Visualization** | tracking_projector | Real-time feedback projection |

---

## Documentation Cross-References

- **Full Architecture:** `docs/core/architecture/INDEX.md`
- **Inference Details:** `docs/core/architecture/inference_pipeline/ARCHITECTURE.md`
- **System Overview:** `docs/core/architecture/system_overview/ARCHITECTURE.md`
- **Tracking Logic:** `docs/core/architecture/tracking_orchestration/ARCHITECTURE.md`
- **Known Issues:** `docs/reference/known-issues.md`

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2026-02-05  
**For:** Russ + Original Engineer Handoff Meeting
