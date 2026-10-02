"""
Exercise 02 - Putting It All Together
======================================
Wire up the complete pipeline: DataLoader → Model → Loss → Backward → Step.
No training loop — just a single forward and backward pass.

Run this script to verify:
    python pipeline.py

Make sure you have completed model.py and dataset.py first.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import TinyNet
from dataset import SimpleImageDataset


# ===========================================================================
# Task 1: Wire the Full Pipeline
# ===========================================================================

# Step 1: Set device
# Determine whether CUDA is available. Create a torch.device that is
# 'cuda' if a GPU exists, otherwise 'cpu'. Print which device you're using.
device = None

# Step 2: Create dataset and dataloader
# Instantiate SimpleImageDataset with root_dir='sample_images/'.
# Apply a transform pipeline: Resize(32,32) → ToTensor() → Normalize().
# Create a DataLoader with batch_size=2, shuffle=True, num_workers=0.
dataset = None
dataloader = None

# Step 3: Instantiate the model
# Create TinyNet. The num_classes should match the number of classes
# in your dataset — use len(dataset.classes) to get this automatically.
# Move the model to the device from Step 1 using .to(device).
model = None

# Step 4: Define loss function and optimizer
# Use nn.CrossEntropyLoss() as the loss function.
# Use torch.optim.Adam() with lr=0.001 as the optimizer.
# Pass model.parameters() to the optimizer.
criterion = None # this is the loss function
optimizer = None

# Step 5: Fetch one batch
# Get one batch of (images, labels) from the dataloader.
# Move both to the device. Tensors are moved with .to(device).
images, labels = None , None

# Step 6: Forward pass
# Pass the images through the model to get output logits.
# Print the output shape — should be [batch_size, num_classes].
outputs = None

# Step 7: Compute loss
# Pass (outputs, labels) to the loss function.
# Print the loss value.
loss = None

# Step 8: Backward pass and optimizer step
# Three calls in order: zero gradients, compute gradients, update weights.


# Step 9: Print predictions
# Get predicted class indices using .argmax(dim=1) on the outputs.
# Print predicted classes and actual labels side by side.
# They will likely not match — the model is untrained.
predictions = None

# ===========================================================================
# Verification
# ===========================================================================
print("\n=== VERIFICATION ===")

def _assert_true(condition, message):
    assert condition, message


def check(label, assertion_fn):
    try:
        assertion_fn()
    except AssertionError as e:
        print(f"[FAIL] {label} - {e}")
        return False
    except NameError as e:
        print(f"[FAIL] {label} (variable not defined: {e})")
        return False
    except Exception as e:
        print(f"[FAIL] {label} (error while checking: {e})")
        return False
    print(f"[PASS] {label}")
    return True


results = []

results.append(check(
    "Step 1 - device matches GPU availability",
    lambda: _assert_true(
        device.type == ("cuda" if torch.cuda.is_available() else "cpu"),
        f"expected device type '{'cuda' if torch.cuda.is_available() else 'cpu'}', got '{device.type}'")))

results.append(check("Step 2 - dataset has at least one class",
                      lambda: _assert_true(len(dataset.classes) > 0, "expected dataset.classes to be non-empty")))
results.append(check("Step 2 - dataloader batch_size is 2",
                      lambda: _assert_true(dataloader.batch_size == 2,
                                            f"expected batch_size 2, got {dataloader.batch_size}")))

results.append(check(
    "Step 3 - model is a TinyNet on `device`",
    lambda: _assert_true(
        isinstance(model, TinyNet) and next(model.parameters()).device.type == device.type,
        "expected model to be a TinyNet moved to `device`")))

results.append(check("Step 4 - criterion is CrossEntropyLoss",
                      lambda: _assert_true(isinstance(criterion, nn.CrossEntropyLoss),
                                            "expected criterion to be nn.CrossEntropyLoss")))
results.append(check(
    "Step 4 - optimizer is Adam with lr=0.001",
    lambda: _assert_true(isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["lr"] == 0.001,
                          "expected an Adam optimizer with lr=0.001")))

results.append(check(
    "Step 5 - images shape is (2, 3, 32, 32) on `device`",
    lambda: _assert_true(images.shape == torch.Size([2, 3, 32, 32]) and images.device.type == device.type,
                          f"expected shape (2, 3, 32, 32) on {device.type}, "
                          f"got {tuple(images.shape)} on {images.device.type}")))
results.append(check(
    "Step 5 - labels has 2 entries on `device`",
    lambda: _assert_true(len(labels) == 2 and labels.device.type == device.type,
                          "expected 2 labels on `device`")))

results.append(check(
    "Step 6 - outputs shape is (2, num_classes)",
    lambda: _assert_true(outputs.shape == torch.Size([2, len(dataset.classes)]),
                          f"expected shape (2, {len(dataset.classes)}), got {tuple(outputs.shape)}")))

results.append(check("Step 7 - loss is a scalar that requires grad",
                      lambda: _assert_true(loss.dim() == 0 and loss.requires_grad,
                                            "expected loss to be a 0-dim tensor with requires_grad=True")))

results.append(check(
    "Step 8 - loss.backward() populated the model gradients",
    lambda: _assert_true(all(p.grad is not None for p in model.parameters()),
                          "expected every model parameter to have a .grad - did you call loss.backward()?")))

results.append(check(
    "Step 9 - predictions has one valid class index per sample",
    lambda: _assert_true(
        predictions.shape == torch.Size([2])
        and bool(((predictions >= 0) & (predictions < len(dataset.classes))).all()),
        f"expected 2 predictions in [0, {len(dataset.classes)})")))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")


# ===========================================================================
# Summary
# ===========================================================================
print("\n--- Pipeline complete. ---")
print("This is the exact sequence that repeats inside every training loop:")
print("  DataLoader → Model → Loss → Backward → Optimizer Step")
print("In the next exercises, you will wrap this in an epoch loop")
print("and add evaluation, checkpointing, and logging.")
