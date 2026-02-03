# PyTorch Tutorial: Cherry Processing Machine - Section 6: Functional Operations

## torch.nn.functional Module

Functional operations provide stateless versions of neural network layers. No parameters stored.

**Import:**

```python
import torch.nn.functional as F
```

## Conv2D: Image Processing

**Reference:** `ai_detector.py:326-327`

```python
import torch.nn.functional as F

# Create 5x5 all-ones kernel
kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]], dtype=np.float32)

kernel_tensor = torch.tensor(
    np.expand_dims(np.expand_dims(kernel, 0), 0),  # Shape: (1, 1, 5, 5)
    device='cuda'
)

# Dilate: Convolve with positive kernel
mask = torch.clamp(
    F.conv2d(mask, kernel_tensor, padding=(1, 1)),
    0, 1
)

# Erode: Convolve with negative kernel
mask = torch.clamp(
    F.conv2d(mask, kernel_tensor * -1, padding=(1, 1)),
    0, 1
)
```

**Conv2D parameters:**
- `input`: `[batch, in_channels, height, width]`
- `weight`: `[out_channels, in_channels, kernel_h, kernel_w]`
- `padding`: `(pad_h, pad_w)` - adds border
- Returns: `[batch, out_channels, out_h, out_w]`

**Dilate/Erode concept:**
- Dilate: Expand bright regions (fill small gaps)
- Erode: Shrink bright regions (remove noise)

## Softmax: Probability Distribution

**Reference:** `ai_detector.py:366`

```python
# Classifications are logits (raw scores)
classifications  # [N, 2] - [[logit_clean, logit_pit], ...]

# Convert to probabilities
probs = F.softmax(classifications, dim=1)

# Result: [N, 2] - [[p_clean, p_pit], ...]
# Each row sums to 1.0
```

**Dim parameter:**
- `dim=0`: Softmax across batches (column-wise)
- `dim=1`: Softmax across classes (row-wise) ← **Use this for classification**

## Clamp: Value Clipping

**Reference:** `ai_detector.py:326`

```python
# Clamp values to [0, 1] range
mask = torch.clamp(F.conv2d(mask, kernel, padding=(1, 1)), 0, 1)
```

**Parameters:**
- `input`: Tensor to clamp
- `min`: Minimum value
- `max`: Maximum value

## Max: Finding Maximum Values

**Reference:** `ai_detector.py:368`

```python
# Get max probability and its index per sample
probs = F.softmax(classifications, dim=1)

conf, classes = torch.max(probs, 1)

# conf: [N] - max probabilities
# classes: [N] - class indices (0 or 1)
```

**torch.max() usage:**
- `torch.max(tensor, dim)`: Returns (values, indices)
- `dim=0`: Max per column
- `dim=1`: Max per row ← **Use this**

## Comparison Operators

**Reference:** `ai_detector.py:376-378, 474-477`

```python
# Greater than or equal
pit_mask = probs[:, 1].ge(0.75)    # True if pit_prob >= 0.75
maybe_mask = probs[:, 1].ge(0.5)    # True if pit_prob >= 0.5
clean_mask = probs[:, 0].ge(0.5)    # True if clean_prob >= 0.5

# Equal to
clean_mask = prediction['labels'].eq(1)  # True where label == 1 (clean)
pit_mask = prediction['labels'].eq(2)    # True where label == 2 (pit)
side_mask = prediction['labels'].eq(3)   # True where label == 3 (side)
```

**Available operators:**
- `.eq(value)`: Equal to
- `.ne(value)`: Not equal
- `.ge(value)`: Greater or equal
- `.gt(value)`: Greater than
- `.le(value)`: Less or equal
- `.lt(value)`: Less than

## torch.where: Conditional Selection

**Reference:** `ai_detector.py:381-383`

```python
# torch.where(condition, value_if_true, value_if_false)

# Set labels based on probability thresholds
prediction['labels'] = torch.where(maybe_mask, 5, prediction['labels'])  # Maybe → 5
prediction['labels'] = torch.where(pit_mask, 2, prediction['labels'])    # Pit → 2
prediction['labels'] = torch.where(clean_mask, 1, prediction['labels'])  # Clean → 1

# Also applies to edge detection
prediction['labels'] = torch.where(prediction['boxes'][:, 0] < 170, 3, prediction['labels'])  # Left edge
prediction['labels'] = torch.where(prediction['boxes'][:, 2] > 2244, 3, prediction['labels'])  # Right edge
```

**Behavior:**
- Creates a new tensor based on condition
- Values can be scalars or tensors
- Shape broadcast if needed

## Functional vs. nn.Module

### Functional (stateless)

```python
import torch.nn.functional as F

# No parameters stored
output = F.conv2d(input, weight, padding=1)
output = F.softmax(input, dim=1)
output = F.max_pool2d(input, kernel_size=2)
```

### nn.Module (stateful)

```python
import torch.nn as nn

# Parameters stored
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv(input)

softmax = nn.Softmax(dim=1)
output = softmax(input)
```

**When to use each:**
- **Functional**: Simple operations, custom weights (like the kernel in this code)
- **nn.Module**: Learnable layers (convolution, linear, etc.)

## Other Useful Functions

```python
# Activation functions
F.relu(x)           # ReLU
F.leaky_relu(x)      # Leaky ReLU
F.sigmoid(x)         # Sigmoid
F.tanh(x)            # Tanh

# Pooling
F.max_pool2d(x, kernel_size=2)      # Max pooling
F.avg_pool2d(x, kernel_size=2)      # Average pooling

# Normalization
F.batch_norm(x, running_mean, running_var, weight, bias)
F.layer_norm(x, normalized_shape)

# Loss functions
F.cross_entropy(input, target)
F.mse_loss(input, target)
F.binary_cross_entropy(input, target)
```

## Next Section

**[07. Training](PYTORCH_07_TRAINING.md)** - Training code examples (Mask R-CNN and ResNet50)
