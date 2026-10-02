"""
Exercise 02 - nn.Module: Building a Network
============================================
Solve each task below. Run this script to verify your solutions:
    python model.py
"""

import torch
import torch.nn as nn


# ===========================================================================
# Task 1: Define TinyNet
# ===========================================================================
# Define a network called TinyNet by subclassing nn.Module.
#
class TinyNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
# Step 1: Define self.features as nn.Sequential with two conv blocks:
#
#         Block 1: Conv2d(3→8, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
#         Block 2: Conv2d(8→16, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)
#
#         Each MaxPool2d(2) halves the spatial dimensions.
#         Input 32×32 → after block 1: 16×16 → after block 2: 8×8.
        self.features = None
# Step 2: Define self.classifier as nn.Sequential with one Linear layer.
#         Input size = 16 channels × 8 × 8 = 1024.
#         Output size = num_classes.
        self.classifier = None
#
# Implementing the forward method:
    def forward(self, x):
#         Pass x through self.features.
#         Flatten: x = x.view(x.size(0), -1)
#           x.size(0) keeps the batch dimension, -1 flattens the rest.
#         Pass through self.classifier.
#         Return the output.
        pass


# ===========================================================================
# Task 2: Forward Pass with Dummy Input
# ===========================================================================
if __name__ == "__main__":
    # Step 1: Instantiate TinyNet with num_classes=5.
    model = None
    #
    # Step 2: Create a dummy input tensor of shape (1, 3, 32, 32).
    #         This represents one RGB 32×32 image.
    dummy_input = None
    #
    # Step 3: Pass it through the model and print:
    #         - Output shape (expected: [1, 5])
    #         - Raw output values (these are logits, not probabilities)
    output = None


    # ===========================================================================
    # Inspect Model Parameters
    # ===========================================================================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nTask 3:")
    print(f"  Model architecture:\n{model}")
    print(f"\n  Total trainable parameters: {total_params}")


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


    EXPECTED_PARAM_COUNT = 6517  # fixed by the architecture spec in Task 1

    results = []

    results.append(check(
        "Task 1 - a fresh TinyNet maps (1, 3, 32, 32) to (1, 5)",
        lambda: _assert_true(
            TinyNet(num_classes=5)(torch.zeros(1, 3, 32, 32)).shape == torch.Size([1, 5]),
            "expected TinyNet(num_classes=5) to map a (1, 3, 32, 32) input to shape (1, 5)")))
    results.append(check(
        "Task 1 - TinyNet has 6517 trainable parameters",
        lambda: _assert_true(
            sum(p.numel() for p in TinyNet(num_classes=5).parameters() if p.requires_grad) == EXPECTED_PARAM_COUNT,
            f"expected {EXPECTED_PARAM_COUNT} trainable parameters for the specified architecture")))

    results.append(check("Task 2 - dummy_input shape is (1, 3, 32, 32)",
                        lambda: _assert_true(dummy_input.shape == torch.Size([1, 3, 32, 32]),
                                                f"expected shape (1, 3, 32, 32), got {tuple(dummy_input.shape)}")))
    results.append(check("Task 2 - output shape is (1, 5)",
                        lambda: _assert_true(output.shape == torch.Size([1, 5]),
                                                f"expected shape (1, 5), got {tuple(output.shape)}")))

    results.append(check("Task 3 - total_params equals 6517",
                        lambda: _assert_true(total_params == EXPECTED_PARAM_COUNT,
                                                f"expected {EXPECTED_PARAM_COUNT}, got {total_params}")))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
