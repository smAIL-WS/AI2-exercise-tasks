"""
Exercise 02 - Loss Functions and Optimizers
============================================
Solve each task below. Run this script to verify your solutions:
    python loss_and_optimizer.py

Note: This script imports TinyNet from model.py. Make sure you have
completed that file first.
"""

import torch
import torch.nn as nn

from model import TinyNet


# ===========================================================================
# Task 1: Compute Loss
# ===========================================================================
# Step 1: Instantiate TinyNet with num_classes=5.
model = None

# Step 2: Create a dummy input of shape (4, 3, 32, 32) — a batch of 4 images.
dummy_input = None

# Step 3: Create dummy target labels as a tensor: [0, 2, 4, 1].
#         These are the "correct" class indices for each image.
targets = None

# Step 4: Define the loss function using nn.CrossEntropyLoss().
criterion = None

# Step 5: Run a forward pass to get outputs, then compute the loss
#         by passing (outputs, targets) to the loss function.
#         Print the loss value using .item() to get a Python float.
outputs=None
loss=None

# ===========================================================================
# Task 2: One Optimizer Step
# ===========================================================================
# Step 1: Define an Adam optimizer using torch.optim.Adam().
#         Pass model.parameters() and lr=0.001.
optimizer = None
# Step 2: Print the loss from Task 1 (before any weight update).
#
# Step 3: Perform the standard update cycle:
#         - Clear old gradients (they accumulate by default), hint: use .zero_grad()
#         - Compute new gradients from the loss (.backward())
#         - Update the weights (.step())
#
# Step 4: Run another forward pass with the same input and compute
#         the new loss. Print it.
#         It should differ from the first — the weights were updated.
outputs_after = None
loss_after = None


# ===========================================================================
# Task 3: Understanding zero_grad
# ===========================================================================
# This task demonstrates why clearing gradients is necessary.
#
# Step 1: Create a fresh TinyNet and a fresh optimizer.

model2 = TinyNet(num_classes=5)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# Step 2: Run a forward pass and backward pass. Pick any parameter
#         from the model (e.g. the first one from model.parameters())
#         and print its .grad.sum() — this is the gradient after one
#         backward pass.

out1 = model2(dummy_input)
loss1 = criterion(out1, targets)
loss1.backward()

# Pick the first conv layer's weight gradient
first_param = list(model2.parameters())[0]
grad_after_first = first_param.grad.clone()

# Step 3: Run another forward + backward WITHOUT clearing gradients
#         first. Print the same parameter's .grad.sum() again.
#         It should be roughly double — gradients accumulated.

out2 = model2(dummy_input)
loss2 = criterion(out2, targets)
loss2.backward()

grad_after_second = first_param.grad.clone()

print("\nUnderstanding zero_grad:")
print(f"  Grad sum after 1st backward: {grad_after_first.sum().item():.6f}")
print(f"  Grad sum after 2nd backward: {grad_after_second.sum().item():.6f}")
print(f"  (Second is ~2x the first because gradients accumulated)")

# This is why every training step must start by clearing gradients.


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

results.append(check("Task 1 - outputs shape is (4, 5)",
                      lambda: _assert_true(outputs.shape == torch.Size([4, 5]),
                                            f"expected shape (4, 5), got {tuple(outputs.shape)}")))
results.append(check("Task 1 - loss is a scalar that requires grad",
                      lambda: _assert_true(loss.dim() == 0 and loss.requires_grad,
                                            "expected loss to be a 0-dim tensor with requires_grad=True")))

results.append(check(
    "Task 2 - optimizer is Adam with lr=0.001",
    lambda: _assert_true(isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["lr"] == 0.001,
                          "expected an Adam optimizer with lr=0.001")))
results.append(check(
    "Task 2 - loss_after differs from the pre-update loss",
    lambda: _assert_true(loss_after.item() != loss.item(),
                          "expected the loss to change after optimizer.step()")))

results.append(check(
    "Task 3 - grad_after_second accumulated on top of grad_after_first",
    lambda: torch.testing.assert_close(grad_after_second, grad_after_first * 2)))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
