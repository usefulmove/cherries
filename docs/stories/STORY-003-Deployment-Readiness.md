# STORY-003: Deployment Readiness

## Goal
Ensure the cherry pit detection model is fully compatible with the production ROS2 system and meets real-time latency requirements before final deployment.

## Acceptance Criteria
- [x] Train a "drop-in replacement" model (Unnormalized) that matches or beats production accuracy (>93%).
- [x] Verify inference latency is <30ms per frame on CPU (matching baseline).
- [ ] Deploy the final model weights to `cherry_system/cherry_detection/resource/`.

## Context Scope
**Write:**
- training/configs/
- training/notebooks/
- training/scripts/benchmark_latency.py

**Read:**
- training/scripts/compare_models.py
- docs/reference/model_experiments.md

## Approach
1. **Unnormalized Training:**
   - The production system feeds raw images (0-255) to the model.
   - Our best model currently expects ImageNet normalization.
   - **Solution:** Train a variant with `augmentation: true` but `normalize: false`. This avoids modifying the ROS2 node code.

2. **Latency Verification:**
   - Previous logs showed a potential regression (115ms vs 25ms).
   - This might be due to the data loader overhead in the evaluation script.
   - **Solution:** Create a pure compute benchmark (`benchmark_latency.py`) to measure raw inference speed on random tensors.

3. **Deployment:**
   - Select the winner (either `resnet50_augmented` with code changes OR `resnet50_unnormalized` without changes).
   - Copy to `cherry_system/...`.

## Notes
- Production Baseline: 92.99% Accuracy, ~16ms inference (CPU).
- **Unnormalized Model Evaluation (2026-01-30):**
    - **Accuracy:** 94.05% (Exceeds target of 93%).
    - **Initial Latency:** ~168ms (Failed target).
    - **Issue:** Weights contained "denormal" values causing CPU slowdown.
    - **Fix:** Applied zeroing of denormal values (< 1e-35).
    - **Final Latency:** ~16.7ms (Matches baseline).
    - **Status:** Ready for deployment.
