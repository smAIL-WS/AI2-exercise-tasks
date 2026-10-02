# Exercise 02: PyTorch Fundamentals

## Overview

This exercise introduces the core building blocks of PyTorch that you will use in every subsequent exercise. **Part I** covers the concepts — your instructor will explain these in class. **Part II** directs you to the scripts where you solve the tasks. No model training is performed — the goal is to understand the mechanics of a deep learning pipeline.

**Estimated time:** 90–120 minutes

**Prerequisites:** Exercise 00 (VS Code + SSH setup), Exercise 01 (Git basics)

**File structure:**
```
exercise_02/
├── README.md              ← You are here
├── tensors.py             ← Tasks for sections 1 & 2
├── autograd.py            ← Tasks for section 3
├── model.py               ← Tasks for section 4
├── loss_and_optimizer.py  ← Tasks for section 5
├── dataset.py             ← Tasks for section 6
├── pipeline.py            ← Tasks for section 7
└── data/
    └── sample_images/
        ├── class_a/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── img3.jpg
        └── class_b/
            ├── img1.jpg
            ├── img2.jpg
            └── img3.jpg
```

---

# Part I: Concepts

---

## 1. Tensors

A tensor is PyTorch's basic data structure — a multi-dimensional array, similar to a NumPy array, but with two critical additions: it can run on a GPU, and it can track gradients for automatic differentiation.

In deep learning, almost everything is a tensor:

- A grayscale image: a 2D tensor of shape `(H, W)` — height, width
- A color image: a 3D tensor of shape `(C, H, W)` — channels, height, width
- A batch of color images: a 4D tensor of shape `(B, C, H, W)` — batch size, channels, height, width

### Key operations

```python
import torch

# Creating tensors
a = torch.zeros(3, 4)          # 3x4 tensor of zeros
b = torch.randn(2, 3)          # 2x3 tensor of random values (normal distribution)
c = torch.tensor([1, 2, 3])    # tensor from a Python list

# Inspecting tensors
print(a.shape)    # torch.Size([3, 4])
print(a.dtype)    # torch.float32 (default)
print(a.device)   # cpu (default)

# Moving to GPU
if torch.cuda.is_available():
    a_gpu = a.to('cuda')
    print(a_gpu.device)   # cuda:0
```

---

## 2. Tensor Operations

### Indexing and Slicing

Tensors support NumPy-style indexing. This is essential for inspecting specific samples, channels, or regions in image data.

```python
x = torch.randn(4, 3, 8, 8)   # batch of 4 RGB 8x8 images

first_image = x[0]             # shape: (3, 8, 8)
red_channel = x[0, 0]          # shape: (8, 8) — first channel of first image
```

### Reshaping

```python
# view / reshape — flatten or restructure
x = torch.randn(2, 3, 4)
y = x.view(2, 12)         # reshape to (2, 12)

# permute — reorder dimensions (e.g. HWC to CHW)
img_hwc = torch.randn(224, 224, 3)    # height, width, channels
img_chw = img_hwc.permute(2, 0, 1)    # channels, height, width — PyTorch convention

# squeeze / unsqueeze — remove or add dimensions of size 1
a = torch.randn(3, 224, 224)          # single image, no batch dimension
a_batched = a.unsqueeze(0)            # shape: (1, 3, 224, 224) — added batch dim
a_back = a_batched.squeeze(0)         # shape: (3, 224, 224) — removed batch dim
```

### Basic Math

```python
a = torch.randn(3, 3)
b = torch.randn(3, 3)

c = a + b              # element-wise addition
d = a * b              # element-wise multiplication
e = a @ b              # matrix multiplication
f = a.sum()            # sum of all elements
g = a.mean(dim=0)      # mean along dimension 0
```

---

## 3. Autograd

Autograd is PyTorch's automatic differentiation engine. When you set `requires_grad=True` on a tensor, PyTorch records every operation performed on it. When you call `.backward()`, it computes all gradients automatically — this is what makes training neural networks possible.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 3 * x    # y = x² + 3x
y.backward()           # computes dy/dx
print(x.grad)          # dy/dx = 2x + 3 = 2(3) + 3 = 9.0
```

### When to Disable Gradient Tracking

During evaluation or inference, you do not need gradients. Disabling them saves memory and computation.

```python
with torch.no_grad():
    output = model(input_tensor)    # no gradient computation
```

---

## 4. nn.Module — Building a Network

`nn.Module` is the base class for all neural networks in PyTorch. You define a network by subclassing it and implementing two methods:

- `__init__`: define the layers
- `forward`: define how data flows through those layers

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # flatten for linear layer
        x = self.classifier(x)
        return x
```

### Common Layers

| Layer | Purpose | Example |
|-------|---------|---------|
| `nn.Conv2d(in, out, kernel)` | Extracts spatial features from images | `nn.Conv2d(3, 16, 3)` |
| `nn.Linear(in, out)` | Fully connected layer | `nn.Linear(256, 10)` |
| `nn.ReLU()` | Activation function | — |
| `nn.BatchNorm2d(channels)` | Stabilizes training | `nn.BatchNorm2d(16)` |
| `nn.MaxPool2d(kernel)` | Downsamples spatial dimensions | `nn.MaxPool2d(2)` |
| `nn.Dropout(p)` | Regularization — randomly zeroes elements | `nn.Dropout(0.5)` |

---

## 5. Loss Functions and Optimizers

### Loss Functions

A loss function measures how far the model's output is from the expected answer. For classification, the standard choice is `nn.CrossEntropyLoss`. It takes raw model outputs (logits) and integer class labels.

```python
criterion = nn.CrossEntropyLoss()

outputs = torch.randn(4, 10)          # batch of 4, 10 classes — raw logits
targets = torch.tensor([3, 0, 7, 1])  # correct class indices

loss = criterion(outputs, targets)
print(loss.item())                     # scalar loss value
```

### Optimizers

An optimizer updates the model's weights based on the computed gradients. The standard update cycle is:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()    # 1. clear old gradients
loss.backward()          # 2. compute new gradients
optimizer.step()         # 3. update weights
```

**Why `zero_grad()`?** PyTorch accumulates gradients by default. Without clearing them, each `.backward()` call adds to the previous gradients rather than replacing them.

---

## 6. Datasets and DataLoaders

In later exercises, you will load images from folders on the server — not from built-in datasets. PyTorch's `Dataset` class provides a standard interface for this. You subclass it and implement:

- `__len__`: return the total number of samples
- `__getitem__`: return one sample (image + label) given an index

### Transforms

Before feeding images to a model, they need to be resized, converted to tensors, and normalized.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),             # converts PIL image to tensor, scales to [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # ImageNet mean
        std=[0.229, 0.224, 0.225]      # ImageNet std
    ),
])
```

### DataLoader

A `DataLoader` wraps a `Dataset` and provides batching, shuffling, and parallel data loading.

```python
from torch.utils.data import DataLoader

dataset = SimpleImageDataset('data/sample_images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

images, labels = next(iter(dataloader))
print(images.shape)    # (2, 3, 32, 32)
```

---

## 7. The Pipeline

Every deep learning project follows the same core sequence:

1. **Load data** → DataLoader yields a batch of `(images, labels)`
2. **Forward pass** → model takes images, produces output logits
3. **Compute loss** → compare logits to true labels
4. **Backward pass** → compute gradients
5. **Update weights** → optimizer adjusts model parameters

```
[DataLoader] → [Model] → [Loss Function] → [Backward] → [Optimizer Step]
     ↑                                                          |
     └──────────────── next batch ────────────────────────────--┘
```

---

# Part II: Exercises

Work through each script in order. Open the file, look for `# TODO` markers, write your solution, and run the script to verify.

---

### Exercise A — Tensors and operations (sections 1 & 2)

**→ Open `tensors.py` and solve the tasks.**

Covers: tensor creation, shape/dtype/device inspection, GPU transfer, indexing, permute (HWC→CHW), squeeze/unsqueeze, reshaping, element-wise and matrix operations.

Verify: `python tensors.py`

---

### Exercise B — Autograd (section 3)

**→ Open `autograd.py` and solve the tasks.**

Covers: creating tensors with `requires_grad`, computing gradients with `.backward()`, manual gradient verification by hand, and disabling tracking with `torch.no_grad()`.

Verify: `python autograd.py`

---

### Exercise C — Building a network (section 4)

**→ Open `model.py` and solve the tasks.**

Covers: subclassing `nn.Module`, defining Conv-ReLU-Pool blocks in `__init__`, implementing `forward()`, passing a dummy input, inspecting parameter count.

Verify: `python model.py`

---

### Exercise D — Loss and optimizer (section 5)

**→ Open `loss_and_optimizer.py` and solve the tasks.**

Requires: `model.py` completed (imports `TinyNet`).

Covers: computing `CrossEntropyLoss`, performing one optimizer step, observing gradient accumulation without `zero_grad()`.

Verify: `python loss_and_optimizer.py`

---

### Exercise E — Dataset and DataLoader (section 6)

**→ Open `dataset.py` and solve the tasks.**

Covers: building a transform pipeline, writing a custom `Dataset` class that scans an ImageFolder directory, creating a `DataLoader`, inspecting batch shapes and normalized pixel values.

Verify: `python dataset.py`

---

### Exercise F — Putting it together (section 7)

**→ Open `pipeline.py` and solve the tasks.**

Requires: `model.py` and `dataset.py` completed (imports both).

Covers: wiring device setup, dataset, dataloader, model, loss, optimizer, and a single forward → backward → step pass into one end-to-end pipeline. This is the skeleton that every training loop in the course builds on.

Verify: `python pipeline.py`

---

## Summary

| Concept | Script | Used in later exercises |
|---------|--------|------------------------|
| Tensors and operations | `tensors.py` | Every exercise |
| Autograd | `autograd.py` | Training and evaluation loops |
| nn.Module | `model.py` | Classification, detection, segmentation |
| Loss and Optimizers | `loss_and_optimizer.py` | Every training exercise |
| Dataset and DataLoader | `dataset.py` | Every exercise with real data |
| Pipeline | `pipeline.py` | The structure of every project |

From Exercise 03 onward, you will work with real datasets, full training loops, and the pytorch-template structure — but the building blocks remain exactly what you practiced here.
