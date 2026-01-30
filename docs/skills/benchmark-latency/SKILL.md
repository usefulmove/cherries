---
name: benchmark-latency
description: Measure model inference speed and inspect weights for performance issues.
---

# Benchmark Latency Skill

This skill allows you to measure the pure compute latency of a model and inspect its weights for potential issues like denormal values that can degrade CPU performance.

## When to use
*   After training a new model, before deployment.
*   If a model feels "slow" during inference.
*   To verify real-time constraints (< 30ms per frame).

## Requirements
*   A trained PyTorch model file (`.pt`).
*   Python environment with `torch`.

## Usage 1: Measure Latency

Run `benchmark_latency.py` to get timing statistics.

```bash
python training/scripts/benchmark_latency.py \
  --model-path <path/to/model.pt> \
  --device <cpu|cuda> \
  [--batch-size 1]
```

### Example

```bash
python training/scripts/benchmark_latency.py \
  --model-path training/experiments/resnet50_augmented_unnormalized/model_best_fixed.pt \
  --device cpu
```

## Usage 2: Inspect Weights (Health Check)

Run `inspect_model.py` to check for "denormal" values and architectural details. Denormal values (extremely small floats) can cause massive CPU slowdowns (10x+).

```bash
python training/scripts/inspect_model.py <path/to/model.pt> [--device cpu]
```

### Interpreting Inspection Output
*   **Denormal Count:** Should be **0**.
    *   If `WARNING: Denormals detected`, run `fix_denormals.py`.
*   **Single Inference Time:** Should match expected baseline (~16ms for ResNet50 on standard CPU).

## Usage 3: Fix Denormals

If `inspect_model.py` reports denormals, use this script to fix the weights.

```bash
python training/scripts/fix_denormals.py <input_model.pt> <output_fixed_model.pt>
```
