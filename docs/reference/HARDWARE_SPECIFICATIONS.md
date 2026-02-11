# Hardware Specifications

## Production System (Edge Inference)

**Status:** GPU model confirmation pending

| Component | Specification | Notes |
|-----------|---------------|-------|
| **GPU** | NVIDIA (model TBD) | Originally planned RTX 3080, actual model unconfirmed |
| **OS** | Ubuntu 22.04 LTS | |
| **PyTorch** | CUDA-enabled | GPU inference supported |
| **Code** | CUDA checks in ai_detector*.py | Falls back to CPU if unavailable |

**Inference Capabilities:**
- Multi-model pipeline (Mask R-CNN + ResNet50 + Stem Detection)
- Real-time processing (baseline: 16ms CPU inference)
- CUDA-accelerated when GPU present

**Known Questions:**
- [ ] Confirm actual GPU model installed in production
- [ ] Measure GPU inference latency for ResNet50 baseline
- [ ] Measure GPU inference latency for ConvNeXt V2

## Training Infrastructure

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Platform** | Google Colab Pro | Model training and experimentation |
| **GPU** | Tesla T4 / A100-SXM4-80GB | Varies by session |
| **Workflow** | Hybrid (local dev + Colab execution) | Code managed locally, training on Colab |

**Why Colab for Training:**
- Local workstation GPU insufficient for model training
- Colab provides access to high-performance GPUs (T4/A100)
- Cost-effective vs. purchasing dedicated training hardware

## Development Workstation

**Purpose:** Code development, relative model comparison

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | Unspecified | Used for relative latency benchmarks |
| **GPU** | Limited/none | Training done on Colab instead |

**Benchmarking Role:**
- CPU latency measurements useful for **relative** model comparison
- ResNet50: 16ms (baseline)
- ConvNeXt V2: 58ms (3.6x slower on CPU)
- Actual production performance will differ (GPU faster)

## Model Latency Targets

| Model Stage | Target | Measurement Device |
|-------------|--------|-------------------|
| Segmentation (Mask R-CNN) | <15ms | Production GPU |
| Classification (ResNet50) | <16ms | Production GPU |
| Total Pipeline | <50ms | Production GPU |

**Note:** Current documented latencies (16ms ResNet50, 58ms ConvNeXt V2) are CPU-based development benchmarks, not production GPU measurements.

## Hardware Evolution

**Legacy System (cherry_system):**
- Cameras: Allied Vision (Mako G-319, Mako G-507)
- Driver: AVT Vimba

**Current System (threading_ws):**
- Cameras: Cognex
- Driver: Cognex SDK
- Improved: HDR multi-layer imaging support
