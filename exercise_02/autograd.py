"""
Exercise 02 - Autograd
======================
Solve each task below. Run this script to verify your solutions:
    python autograd.py
"""

import torch


# ===========================================================================
# Task 1: Basic Autograd
# ===========================================================================
# Step 1: Create a tensor x with value 4.0 and requires_grad=True.
#         This tells PyTorch to track all operations on x.
#
# Step 2: Compute y = 3 * x^3 - 2 * x^2 + x
#         Use ** for exponentiation.
#
# Step 3: Call y.backward() to compute the gradient dy/dx.
#
# Step 4: Print x.grad (use .item()) — this holds the computed gradient.


# ===========================================================================
# Task 2: Manual Gradient Verification
# ===========================================================================
# The purpose of this task is to verify what autograd computed in Task 1.
#
# Given: y = 3x³ - 2x² + x
#
# Step 1: Derive dy/dx on paper using basic differentiation rules.
#         (Power rule: d/dx[xⁿ] = n·xⁿ⁻¹)
#         Write your derivative as a comment below.
#
# dy/dx = ???

# Step 2: Substitute x = 4.0 into your derivative and compute the
#         numerical value by hand. Write it as a comment.
#
# dy/dx at x = 4.0 = ???

# Step 3: Compare your manual result with x.grad from Task 1.
#         Do they match?
#
# Answer:


# ===========================================================================
# Task 3: Second Function — Full Verification
# ===========================================================================
# Define f(x) = 5x² - 4x + 7, and use x = 2.0.
#
# Step 1: Derive df/dx on paper. Write as a comment.
#
# df/dx = ???
# df/dx at x = 2.0 = ???
#
# Step 2: Create a new tensor x2 with value 2.0 and requires_grad=True.
#         Compute f(x2), call .backward(), and print the gradient.
#         Verify it matches your manual calculation.
#
# Important: Each tensor can only have .backward() called once.
# You need a fresh tensor for this task, not the one from Task 1.
x2 = None


# ===========================================================================
# Task 4: torch.no_grad()
# ===========================================================================
# Step 1: Create a tensor x3 with value 5.0 and requires_grad=True.
x3 = None

# Step 2: Inside a "with torch.no_grad():" block, compute y_no_grad = x3 * 2.
#         Print y_no_grad.requires_grad — should be False.
y_no_grad = None

# Step 3: Outside the block, compute z_with_grad = x3 * 2.
#         Print z_with_grad.requires_grad — should be True.
#
# This demonstrates how no_grad() prevents gradient tracking.
# You will use this during model evaluation in later exercises
# to save memory and computation.
z_with_grad = None


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


def _assert_leaf(tensor, name, value):
    _assert_true(isinstance(tensor, torch.Tensor), f"expected {name} to be a tensor, got {type(tensor).__name__}")
    _assert_true(tensor.item() == value,
                 f"expected {name} = {value}, got {tensor.item()} - "
                 f"use a separate tensor per task (x, x2, x3) instead of overwriting one")
    _assert_true(tensor.requires_grad, f"expected {name} to be created with requires_grad=True")


def _assert_grad(tensor, name, expected):
    _assert_true(tensor.grad is not None, f"{name}.grad is None - did you call .backward()?")
    torch.testing.assert_close(tensor.grad, torch.tensor(expected),
                               msg=f"expected {name}.grad = {expected}, got {tensor.grad.item()}")


results = []

results.append(check("Task 1 - x is a tensor with value 4.0 and requires_grad=True",
                     lambda: _assert_leaf(x, "x", 4.0)))
results.append(check("Task 1 - x.grad matches dy/dx = 9x^2 - 4x + 1 at x=4.0 (129.0)",
                     lambda: _assert_grad(x, "x", 129.0)))

results.append(check("Task 3 - x2 is a tensor with value 2.0 and requires_grad=True",
                     lambda: _assert_leaf(x2, "x2", 2.0)))
results.append(check("Task 3 - x2.grad matches df/dx = 10x - 4 at x=2.0 (16.0)",
                     lambda: _assert_grad(x2, "x2", 16.0)))

results.append(check("Task 4 - x3 is a tensor with value 5.0 and requires_grad=True",
                     lambda: _assert_leaf(x3, "x3", 5.0)))
results.append(check("Task 4 - y_no_grad.requires_grad is False (computed inside torch.no_grad())",
                     lambda: _assert_true(y_no_grad.requires_grad is False,
                                          "expected y_no_grad.requires_grad to be False")))
results.append(check("Task 4 - z_with_grad.requires_grad is True (computed outside torch.no_grad())",
                     lambda: _assert_true(z_with_grad.requires_grad is True,
                                          "expected z_with_grad.requires_grad to be True")))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
