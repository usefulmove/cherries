# State-of-the-Art Research: Cherry Pit Classification

**Date:** 2026-02-06  
**Phase:** Phase 1 - Initial Research  
**Status:** In Progress  

---

## 1. Architecture Research Findings

### 1.1 ConvNeXt (2022) - "A ConvNet for the 2020s"

**Paper:** [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)  
**Code:** https://github.com/facebookresearch/ConvNeXt  
**PyTorch Hub:** ✅ Available (`torchvision.models.convnext`)

**Key Findings:**
- Modernized pure ConvNet that competes with Vision Transformers
- Achieves 87.8% ImageNet top-1 accuracy
- Outperforms Swin Transformers on COCO detection
- Maintains simplicity and efficiency of standard ConvNets

**Modernization Techniques from ResNet:**
1. **Macro Design:**
   - Changed stage compute ratio from [3,4,6,3] to [3,3,9,3]
   - Changed stem from 7×7 stride-2 to patchify (4×4 stride-4)
   - Similar to Swin Transformer stem design

2. **ResNeXt-ify:**
   - Uses grouped convolutions (depthwise convolution)
   - Reduces FLOPs while maintaining accuracy

3. **Inverted Bottleneck:**
   - ResNet: [1×1, 3×3, 1×1] with residual after 1×1
   - ConvNeXt: [1×1, 3×3, 1×1] with residual after bottleneck
   - Similar to MobileNetV2 design

4. **Large Kernel Sizes:**
   - Uses 7×7 depthwise convolutions (vs 3×3 in ResNet)
   - Captures larger receptive fields like transformers

5. **Micro Design:**
   - Replaces ReLU with GELU activation
   - Fewer activation functions (like transformers)
   - Fewer normalization layers
   - Layer Normalization instead of BatchNorm
   - Separate downsampling layers

**Variants & Specifications:**

| Model | Params | FLOPs | ImageNet Top-1 | Model Size (est.) |
|-------|--------|-------|----------------|-------------------|
| ConvNeXt-Tiny | 28M | 4.5B | 82.1% | ~110MB |
| ConvNeXt-Small | 50M | 8.7B | 83.1% | ~200MB |
| ConvNeXt-Base | 89M | 15.4B | 83.8% | ~356MB |

**Relevance to Cherry Classification:**
- **Pros:**
  - Pure ConvNet - no architectural paradigm shift
  - Excellent PyTorch support and pretrained weights
  - Better accuracy/parameter tradeoff than ResNet
  - Modernized design lessons can be applied even if full model is too large
  
- **Cons:**
  - ConvNeXt-Base is larger than our 50M parameter limit (89M params)
  - ConvNeXt-Tiny is comparable to ResNet50 (28M vs 25.6M)
  - May be slower on CPU due to depthwise convolutions

**Recommendation:** Test ConvNeXt-Tiny as alternative to ResNet50. Comparable parameters (28M vs 25.6M), but modernized design may improve accuracy.

---

### 1.2 EfficientNet (2019) - Compound Scaling

**Paper:** [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)  
**Code:** https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet  
**PyTorch:** ✅ Available (`torchvision.models.efficientnet`)

**Key Findings:**
- Systematically studies model scaling (depth, width, resolution)
- Introduces compound scaling: uniformly scales all dimensions
- EfficientNet-B7 achieves 84.3% ImageNet top-1 accuracy
- 8.4x smaller and 6.1x faster than previous SOTA

**Compound Scaling Formula:**
```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ

s.t. α · β² · γ² ≈ 2
```

Where φ is a compound coefficient determined by available resources.

**EfficientNet Variants:**

| Model | Params | FLOPs | ImageNet Top-1 | Mobile CPU Latency |
|-------|--------|-------|----------------|-------------------|
| EfficientNet-B0 | 5.3M | 0.39B | 77.3% | 29ms |
| EfficientNet-B1 | 7.8M | 0.70B | 79.1% | 38ms |
| EfficientNet-B2 | 9.2M | 1.0B | 80.1% | 47ms |
| EfficientNet-B3 | 12M | 1.8B | 81.6% | 67ms |
| EfficientNet-B4 | 19M | 4.2B | 82.9% | 100ms |
| EfficientNet-B5 | 30M | 9.9B | 83.6% | 143ms |
| EfficientNet-B6 | 43M | 19.0B | 84.0% | 201ms |
| EfficientNet-B7 | 66M | 37.0B | 84.3% | 299ms |

**Our Previous Experience:**
- Tested EfficientNet-B0 in Experiment Set 1
- Result: 92.66% accuracy (vs 94.05% ResNet50)
- Used with normalization (may have affected results)
- **Caveat:** Production uses unnormalized (0-255) input, which we didn't test with EfficientNet

**Relevance to Cherry Classification:**
- **Pros:**
  - Excellent accuracy/parameter tradeoff
  - B4 (19M params) is within our size constraints
  - Proven on transfer learning tasks
  - PyTorch pretrained weights available
  
- **Cons:**
  - Our B0 test underperformed (but used normalized input)
  - Compound scaling may not suit small datasets
  - Mobile latency measured on Pixel 1 - our CPU may differ

**Recommendation:** Retest EfficientNet with unnormalized training. Try B2-B4 range. The compound scaling principle is solid; our B0 test may have failed due to normalization mismatch.

---

### 1.3 MobileViT (2021) - Hybrid CNN-Transformer

**Paper:** [arXiv:2110.02178](https://arxiv.org/abs/2110.02178)  
**Code:** https://github.com/apple/ml-cvnets  
**PyTorch:** ⚠️ Via `timm` or separate implementation

**Key Findings:**
- Combines CNN spatial inductive bias with Transformer global processing
- Treats transformers "as convolutions" for efficiency
- 6M parameters achieve 78.4% ImageNet accuracy
- Outperforms MobileNetV3 by 3.2% and DeIT by 6.2%

**Architecture Innovation:**
- Uses local-global-local (LGL) block:
  1. Local representation (CNN layers)
  2. Global processing (Transformer blocks)
  3. Fusion (combines local + global)
- MobileViT block replaces standard convolution
- Maintains spatial resolution throughout

**MobileViT Variants:**

| Model | Params | Top-1 Accuracy | Notes |
|-------|--------|----------------|-------|
| MobileViT-XS | 2.3M | 74.8% | Smallest variant |
| MobileViT-S | 5.6M | 78.4% | Default mobile size |
| MobileViT-S (re-trained) | 5.6M | 79.3% | Better training recipe |

**Relevance to Cherry Classification:**
- **Pros:**
  - Very small (5.6M params vs ResNet50's 25.6M)
  - Hybrid design may capture both local details and global context
  - Designed for mobile = good CPU performance
  - Novel approach worth testing
  
- **Cons:**
  - Lower ImageNet accuracy (78.4% vs 87.8% ConvNeXt)
  - May struggle with small dataset
  - PyTorch implementation less mature than ConvNeXt
  - Two-stage training methodology unclear for our use case

**Recommendation:** Interesting for speed-optimized deployment, but may not beat ResNet50 accuracy. Worth testing if we want a sub-10ms model that outperforms ResNet18 (91.92%).

---

### 1.4 Other Notable Architectures

#### EfficientNetV2 (2021)
- Improvements: progressive learning, Fused-MBConv blocks
- Better training speed and parameter efficiency
- **Status:** Available in PyTorch, worth investigating

#### RepVGG (2021)
- Reparameterization: training-time multi-branch, inference-time single-path
- Better accuracy/speed tradeoff than ResNet
- **Status:** Simple ResNet-like architecture, easy to test

#### Vision Transformers (ViT)
- Pure transformer approach
- **Status:** Likely too heavy for our latency budget without significant modification
- **Verdict:** Skip - ConvNeXt already achieves transformer-level performance with ConvNet efficiency

---

## 2. Three-Class Classification Strategies

### 2.1 Problem Statement

**Current Situation:**
- Training data: 2-class (clean/pit)
- Production requirement: 3-class (clean/pit/maybe)
- Current production: Uses problematic 2-stage training methodology

**Goal:** Find robust approach to generate 3-class output from 2-class training.

---

### 2.2 Option A: Post-hoc Threshold Calibration (Recommended)

**Approach:**
1. Train standard 2-class classifier (clean vs pit)
2. Apply post-training thresholding to create "maybe" class:
   - p(pit) ≥ 0.75 → Pit
   - 0.5 ≤ p(pit) < 0.75 → Maybe
   - p(pit) < 0.5 → Clean
3. Optimize thresholds on validation set

**Advantages:**
- No architectural changes required
- Uses existing best model (94.05% ResNet50)
- Thresholds can be tuned for business requirements
- Easy to implement and iterate

**Disadvantages:**
- "Maybe" class not explicitly trained
- Model doesn't learn from uncertain examples
- Threshold selection is heuristic

**Implementation:**
- Use existing 94.05% ResNet50 model
- Run systematic threshold search on validation set
- Optimize for: pit recall (safety), clean precision (quality), maybe rate (manual review load)

**Evidence:**
- This is essentially what production does currently
- Current thresholds: 0.75 (pit), 0.5 (maybe), 0.5 (clean)
- We can optimize these systematically

---

### 2.3 Option B: Synthetic "Maybe" Data Generation

**Approach:**
1. Generate ambiguous examples between clean and pit
2. Use Mixup or CutMix to interpolate between classes
3. Label synthetic samples as "maybe"
4. Train true 3-class model

**Data Generation Methods:**

**Mixup (2017):**
```
x_mix = λx_clean + (1-λ)x_pit
y_mix = λy_clean + (1-λ)y_pit
```
Where λ ∈ [0,1]. When λ ∈ [0.4, 0.6], label as "maybe".

**CutMix (2019):**
- Cut and paste patches between clean and pit images
- Labels weighted by patch area
- More realistic than Mixup

**Advantages:**
- True 3-class training
- Model learns uncertainty boundaries explicitly
- Can generate unlimited "maybe" training data

**Disadvantages:**
- Synthetic data may not match real ambiguity
- Risk of model learning artifacts from generation
- Requires careful validation on real edge cases

**Implementation Complexity:**
- Medium: Need to modify data loader for Mixup/CutMix
- Need to define "maybe" threshold for λ values
- Training loop modifications required

---

### 2.4 Option C: Soft Labeling with Uncertainty

**Approach:**
1. Use label smoothing on clean/pit classes
2. Train with soft labels: clean=[0.9, 0.1], pit=[0.1, 0.9]
3. Post-process predictions into 3 classes based on confidence
4. Low confidence predictions → "maybe"

**Label Smoothing:**
```python
# Instead of hard labels [1, 0] and [0, 1]
clean_label = [0.9, 0.1]  # or [0.95, 0.05]
pit_label = [0.1, 0.9]    # or [0.05, 0.95]
```

**Advantages:**
- Regularizes model, prevents overconfidence
- Naturally produces calibrated probabilities
- Simple modification to existing training

**Disadvantages:**
- Smoothing parameter (α) needs tuning
- May slightly reduce peak accuracy
- Still requires threshold selection

**Evidence:**
- Label smoothing widely used in ImageNet training
- α=0.1 typically used (i.e., [0.9, 0.1] instead of [1.0, 0.0])

---

### 2.5 Option D: Evidential Deep Learning

**Approach:**
1. Replace softmax with evidential layer (Dirichlet distribution)
2. Model explicitly outputs: class probabilities + uncertainty
3. High uncertainty → "maybe" class
4. Train with evidential loss function

**Key Paper:** "Evidential Deep Learning to Quantify Classification Uncertainty" (2018)

**Advantages:**
- Principled uncertainty quantification
- Model learns when it doesn't know
- Natural "maybe" class from high uncertainty

**Disadvantages:**
- Significant architectural change
- New loss function to implement
- Limited PyTorch implementations available
- May be overkill for our use case

**Recommendation:** Skip unless Options A-C fail. High complexity for uncertain gain.

---

### 2.6 Three-Class Strategy Comparison

| Strategy | Complexity | Data Requirement | Risk | Recommendation |
|----------|-----------|-----------------|------|----------------|
| **Threshold Calibration** | Low | Existing model | Low | **Start here** |
| **Synthetic Data (Mixup)** | Medium | 2-class + synthesis | Medium | If thresholds insufficient |
| **Soft Labeling** | Low | 2-class | Low | Combine with threshold calibration |
| **Evidential DL** | High | 2-class | High | Last resort only |

**Recommended Path:**
1. **Phase 3A:** Optimize thresholds on existing 94.05% model
2. **Phase 3B:** If needed, train with soft labels + threshold calibration
3. **Phase 3C:** If still insufficient, generate synthetic "maybe" data with Mixup

---

## 3. Architecture Recommendations (Preliminary)

Based on initial research, here are the top candidates for experimentation:

### Tier 1: High Priority (Test First)

**1. ConvNeXt-Tiny**
- Params: 28M (similar to ResNet50)
- Expected latency: ~15-20ms (comparable to ResNet50)
- Rationale: Modernized ResNet, better design principles
- Implementation: ✅ PyTorch native support

**2. EfficientNet-B2/B3 (Unnormalized)**
- Params: 9.2M-12M (smaller than ResNet50)
- Expected latency: ~10-15ms
- Rationale: Retest with proper unnormalized training
- Implementation: ✅ PyTorch native support
- **Note:** Our previous B0 test used normalized input, which may explain lower accuracy

**3. ResNet50 with Attention (CBAM/SE)**
- Params: ~26-28M (minimal increase)
- Expected latency: ~18-22ms
- Rationale: Minimal architectural change, proven technique
- Implementation: Easy - add attention modules to existing ResNet

### Tier 2: Medium Priority (If Tier 1 Underperforms)

**4. ConvNeXt-Small**
- Params: 50M (at our limit)
- Expected latency: ~25-30ms (at budget limit)
- Rationale: More powerful than Tiny, but may be too slow

**5. MobileViT-S**
- Params: 5.6M (very small)
- Expected latency: ~10-12ms
- Rationale: Hybrid CNN-Transformer, novel approach
- **Note:** May struggle with small dataset

### Tier 3: Low Priority (Novelty/Backup)

**6. RepVGG-A2**
- Rationale: Reparameterization for speed, easy implementation
- **Status:** Test if we need pure speed optimization

**7. EfficientNet-B4**
- Params: 19M
- Rationale: Larger EfficientNet variant
- **Note:** Previous B0 result makes this lower confidence

---

## 4. Training Methodology Enhancements

### 4.1 Test-Time Augmentation (TTA)

**Approach:**
- Run inference on multiple augmented versions of test image
- Average predictions
- Reduces variance, improves calibration

**Feasibility:**
- 3x TTA = 3x latency (~48ms vs 16ms)
- Still within 30ms budget? **No** (48ms > 30ms)
- **Verdict:** Skip - exceeds latency budget

### 4.2 Label Smoothing

**Approach:**
- Use soft labels: [0.9, 0.1] instead of [1.0, 0.0]
- Reduces overconfidence
- May improve 3-class separation

**Recommendation:** 
- ✅ Use in all future training runs
- α=0.1 standard value
- Minimal implementation effort

### 4.3 Knowledge Distillation

**Approach:**
1. Train large teacher model (ResNet101, ~45M params)
2. Distill to student (ResNet50 or ResNet18)
3. Student may outperform direct training

**Feasibility:**
- Requires training teacher first (additional compute)
- Teacher may exceed latency budget but improves student
- Colab Pro can handle ResNet101 training

**Recommendation:**
- Consider for Phase 3 if accuracy plateaued
- Test distilling to ResNet18 for speed + accuracy

---

## 5. Open Questions & Next Research Steps

### 5.1 Architecture Questions

1. **ConvNeXt on Small Datasets:**
   - Does ConvNeXt transfer well to small datasets (like our ~5k samples)?
   - Need to check literature for transfer learning results

2. **EfficientNet Unnormalized:**
   - Why did B0 perform poorly (92.66% vs 94.05%)?
   - Was it the normalization mismatch?
   - Need to verify EfficientNet can handle 0-255 input range

3. **Attention Module Performance:**
   - SE (Squeeze-Excitation) vs CBAM vs ECA
   - Which adds least latency for most accuracy gain?

### 5.2 Three-Class Questions

1. **Threshold Optimization Strategy:**
   - What metric to optimize? (pit recall, overall accuracy, maybe rate?)
   - Do we have business constraints on false negative rate?

2. **"Maybe" Validation:**
   - Without labeled "maybe" data, how do we validate 3-class performance?
   - Can we use prediction confidence distribution?

3. **Synthetic Data Quality:**
   - Will Mixup generate realistic ambiguous examples?
   - Risk of model learning interpolation artifacts

### 5.3 Data Questions

1. **Dataset Size:**
   - Exact count of clean vs pit samples
   - Is current data sufficient for modern architectures?

2. **Data Augmentation:**
   - Current: rotation, affine transforms
   - Should we add: color jitter, random erasing, autoaugment?

---

## 6. Preliminary Experimental Design

Based on this research, here are the proposed experiments for Phase 3:

### Experiment 1: Threshold Optimization (Baseline)
**Goal:** Optimize "maybe" thresholds on existing 94.05% model  
**Approach:** Grid search over threshold space  
**Metrics:** Pit recall, clean precision, maybe rate  
**Duration:** 1 day (no training required)  
**Priority:** HIGH - Immediate value

### Experiment 2: ConvNeXt-Tiny 3-Class
**Goal:** Test modernized architecture  
**Approach:** Train ConvNeXt-Tiny with unnormalized + augmentation  
**Variations:** 2-class baseline, 2-class + soft labels  
**Metrics:** Accuracy vs ResNet50, latency on CPU  
**Duration:** 1 day training + evaluation  
**Priority:** HIGH - Top architecture candidate

### Experiment 3: EfficientNet-B2 (Corrected)
**Goal:** Retest EfficientNet with unnormalized training  
**Approach:** Train EfficientNet-B2 (9.2M params)  
**Note:** Previous B0 test used normalized input  
**Metrics:** Accuracy, compare to 94.05% baseline  
**Duration:** 1 day  
**Priority:** HIGH - Previous failure may be due to setup

### Experiment 4: ResNet50 + SE Blocks
**Goal:** Quick accuracy boost with minimal changes  
**Approach:** Add Squeeze-Excitation attention to ResNet50  
**Metrics:** Accuracy gain vs latency cost  
**Duration:** 1 day  
**Priority:** MEDIUM - Low effort, moderate gain potential

### Experiment 5: Label Smoothing + Synthetic "Maybe"
**Goal:** Train true 3-class model  
**Approach:** Use Mixup to generate synthetic "maybe" samples  
**Variations:** Mixup ratios (λ ∈ [0.4, 0.6] for maybe)  
**Metrics:** 3-class accuracy, per-class performance  
**Duration:** 2 days (need to tune Mixup parameters)  
**Priority:** MEDIUM - If threshold optimization insufficient

---

## 7. References

### Architecture Papers
1. **ConvNeXt:** Liu et al., "A ConvNet for the 2020s," CVPR 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
2. **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling," ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
3. **MobileViT:** Mehta & Rastegari, "MobileViT: Light-weight Vision Transformer," ICLR 2022. [arXiv:2110.02178](https://arxiv.org/abs/2110.02178)
4. **ResNet:** He et al., "Deep Residual Learning for Image Recognition," CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

### Training & Regularization
5. **Mixup:** Zhang et al., "Mixup: Beyond Empirical Risk Minimization," ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
6. **CutMix:** Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers," ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
7. **Label Smoothing:** Szegedy et al., "Rethinking the Inception Architecture," CVPR 2016.
8. **Squeeze-Excitation:** Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)

### Uncertainty & 3-Class
9. **Evidential Deep Learning:** Sensoy et al., "Evidential Deep Learning," AAAI 2018. [arXiv:1806.01768](https://arxiv.org/abs/1806.01768)
10. **Temperature Scaling:** Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.

---

## 8. Action Items

- [ ] Verify ConvNeXt pretrained weights available in PyTorch
- [ ] Verify EfficientNet-B2/B3 available and can accept unnormalized input
- [ ] Check if SE-ResNet50 available in torchvision
- [ ] Inventory training data (exact clean/pit counts)
- [ ] Document current production threshold logic more precisely
- [ ] Implement threshold optimization script
- [ ] Create Mixup data loader for synthetic "maybe" generation
- [ ] Prepare Colab notebooks for each experiment

---

**Next Update:** After completing additional research on attention mechanisms (CBAM vs SE) and verifying PyTorch model availability.
