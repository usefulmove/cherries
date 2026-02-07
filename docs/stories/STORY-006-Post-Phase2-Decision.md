# STORY-006: Post-Phase2 Decision - Accuracy vs Latency Tradeoff

## Goal

Make deployment decision after Phase 2 experiments revealed ConvNeXt V2 achieves superior accuracy (94.21%) but fails latency requirements (58ms vs 30ms target). Determine optimal path forward given the accuracy-latency tradeoff.

## Current State

### Best Models Comparison

| Model | Accuracy | Latency (CPU) | Size | Status |
|-------|----------|---------------|------|--------|
| ResNet50 (production) | 92.99% | ~16ms | 90MB | Current deployed |
| **ResNet50 (Phase 1)** | **94.05%** | ~16ms | 90MB | **Best practical** |
| ConvNeXt V2-Tiny | **94.21%** | **~58ms** | 111MB | Too slow |
| ResNet18 | 91.92% | ~8-10ms | 43MB | Speed candidate |

### Critical Finding
ConvNeXt V2's 0.16% accuracy improvement costs **3.6x latency increase** - not viable for real-time conveyor inspection.

## Decision Options

### Option A: Optimize ConvNeXt V2 (Pursue 94.21% accuracy)
**Approach:** Apply model optimization techniques to reduce latency
- **Quantization** (INT8): Potential 2-4x speedup, minimal accuracy loss
- **Pruning**: Remove redundant weights/parameters
- **ONNX Runtime**: Optimize inference engine
- **Knowledge distillation**: Train smaller student model

**Pros:**
- Keep best accuracy (94.21%)
- Modern architecture with FCMAE pre-training
- Future-proof for more complex defects

**Cons:**
- Requires significant engineering effort
- Uncertain if can reach <30ms target
- Risk of accuracy degradation during optimization
- Needs testing on production hardware

**Time estimate:** 2-3 days exploration

---

### Option B: SE-ResNet50 or Architecture Variants (Target ~94% + better speed)
**Approach:** Try squeeze-and-excitation or other ResNet variants
- **SE-ResNet50**: Channel attention, ~25M params, potentially better accuracy
- **ResNeXt-50**: Cardinality dimension, grouped convolutions
- **ECA-Net**: Efficient channel attention, minimal overhead

**Pros:**
- ResNet family = proven fast inference
- Minimal architecture changes from current best
- Can reuse existing training pipeline
- Likely maintains ~16ms latency

**Cons:**
- May not beat 94.05% significantly
- Another round of experiments needed
- No guarantee of success

**Time estimate:** 1-2 days training

---

### Option C: Accept 94.05% with ResNet50 + Threshold Optimization
**Approach:** Deploy current best model and focus on operational improvements
- Use ResNet50 unnormalized (94.05%)
- Run threshold optimization for 3-class deployment
- Optimize "maybe" category handling
- Improve pit recall to ≥99% for food safety

**Pros:**
- **Immediate deployment** - proven latency
- ResNet50 well-understood in production
- Focus shifts to safety/operational improvements
- Can revisit ConvNeXt optimization later

**Cons:**
- Leave 0.16% accuracy on the table
- ConvNeXt theoretically better for edge cases

**Time estimate:** 1 day for threshold optimization

---

### Option D: ResNet18 for Speed-Critical Scenarios (91.92%)
**Approach:** Deploy ResNet18 where latency is paramount
- **91.92% accuracy** (2% drop from ConvNeXt)
- **~8-10ms latency** (50% faster than ResNet50)
- **43MB model** (2x smaller)

**Pros:**
- Fastest inference option
- Smallest model size
- Good for high-throughput scenarios

**Cons:**
- **Pit recall only 89.58%** - food safety concern
- Significant accuracy drop vs 94.05%

**Time estimate:** Already trained, needs threshold optimization

---

### Option E: Hybrid / Multi-Model Approach
**Approach:** Use multiple models for different scenarios
- **Primary:** ResNet50 (94.05%) for standard operation
- **Fast path:** ResNet18 (91.92%) for high-speed mode
- **Research:** Continue ConvNeXt optimization in parallel

**Pros:**
- Flexibility for different operational modes
- Can A/B test in production
- Doesn't block on single decision

**Cons:**
- Complexity in deployment and maintenance
- More models to manage and version

## Acceptance Criteria

- [ ] Decision made on deployment path
- [ ] Documentation updated with decision rationale
- [ ] Next actions defined and assigned
- [ ] Timeline established for implementation

## Context Scope

**Write:**
- docs/reference/MODEL_EXPERIMENTS.md (decision notes)
- docs/reference/LESSONS.md (if applicable)
- docs/stories/STORY-006-Post-Phase2-Decision.md (this file)

**Read:**
- docs/reference/MODEL_EXPERIMENTS.md (Experiment Set 4)
- docs/reference/PHASE2_IMPLEMENTATION_SUMMARY.md
- docs/reference/LESSONS.md
- temp-phase2-experiments/evaluation_results.json

**Exclude:**
- threading_ws/ (production system - not modifying)
- cherry_system/ (legacy - reference only)

## Discussion Points

1. **Latency requirements:** Is 30ms a hard constraint or can we negotiate 40-50ms for better accuracy?

2. **Production hardware:** What's the actual CPU in deployment? Our test may not match.

3. **Business priorities:** 
   - Is 94.05% "good enough" to ship?
   - Or is 94.21% worth the optimization effort?
   - Is speed (ResNet18) ever more important than accuracy?

4. **Food safety:** Current pit recall is 92.58% (ConvNeXt) - threshold optimization can improve this regardless of model choice.

5. **Resources:** How much time to invest vs. moving to other priorities?

## References

- Phase 2 results: [Model Experiments](../reference/MODEL_EXPERIMENTS.md) (Experiment Set 4)
- Summary: [Phase 2 Implementation](../reference/PHASE2_IMPLEMENTATION_SUMMARY.md)
- Session log: [Phase 2 Complete](../logs/session-2026-02-06-phase2-complete.md)
- Evaluation data: `temp-phase2-experiments/evaluation_results.json`
