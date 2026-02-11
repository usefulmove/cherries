# Experimental Models

Models that achieved high accuracy but are not yet deployed to production.

## ConvNeXt V2-Tiny (FCMAE)

**Status:** Research complete, pending GPU benchmarking  
**Accuracy:** 94.21% (2-class: clean/pit)  
**Date:** 2026-02-06  
**Location:** `convnextv2/model_best.pt`

### Performance
- **CPU Latency:** 58ms (measured on dev workstation)
- **GPU Latency:** TBD - needs production GPU benchmark
- **Target:** Maintain ~16ms throughput (baseline) on production hardware

### Training Details
- Architecture: ConvNeXt V2-Tiny with FCMAE pre-training (timm)
- Parameters: 28.6M
- Model Size: 111 MB
- Best Epoch: 22/30
- F1 Score: 0.9416

### Per-Class Performance (Epoch 22)
| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| cherry_clean | 93.76% | 95.61% | 94.67% |
| cherry_pit | 94.76% | 92.58% | 93.66% |

### Confusion Matrix
```
[[631  29]
 [ 42 524]]
```
- True Negatives (correct clean): 631
- False Positives (clean→pit): 29
- False Negatives (pit→clean): 42
- True Positives (correct pit): 524

### Next Steps
1. Benchmark on production GPU hardware
2. Test FP16/INT8 quantization if needed
3. Deploy if GPU latency is acceptable (comparable to baseline)

### References
- Training notebook: `training/notebooks/archive/colab_phase2_experiments.ipynb`
- Results: `docs/reference/MODEL_EXPERIMENTS.md` (Experiment Set 4)
- Session log: `docs/logs/session-2026-02-06-phase2-complete.md`
- CPU evaluation: `docs/reference/convnextv2_cpu_evaluation_results.json`
