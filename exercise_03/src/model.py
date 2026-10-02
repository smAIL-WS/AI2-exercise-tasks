"""
Exercise 03 - CNN Architecture
================================
Build a CNN from scratch for plant disease classification.

Architecture:
    3 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
    followed by AdaptiveAvgPool2d and a classification head.

Run this script to verify:
    python src/model.py
"""

import torch
import torch.nn as nn


class PlantCNN(nn.Module):
    """
    A simple CNN for image classification.

    Args:
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability in the classifier head.
    """

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        # TODO: Define self.features as nn.Sequential with three conv blocks.
        #
        #   Each block follows the same pattern:
        #       Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)
        #
        #   Block 1: 3 input channels → 32 output channels, kernel_size=3, padding=1
        #   Block 2: 32 → 64
        #   Block 3: 64 → 128
        #
        #   padding=1 with kernel_size=3 keeps spatial dimensions unchanged
        #   before pooling. Each MaxPool2d(2) halves the dimensions.
        #   BatchNorm2d takes the number of channels as its argument.
        self.features = None

        # TODO: Define self.pool as nn.AdaptiveAvgPool2d(1).
        #   This reduces any spatial size to 1×1, so the classifier
        #   always receives 128 features regardless of input image size.
        self.pool = None

        # TODO: Define self.classifier as nn.Sequential:
        #   Linear(128, 64) → ReLU → Dropout(dropout) → Linear(64, num_classes)
        #
        #   The input is 128 because self.pool outputs 128 channels × 1 × 1.
        self.classifier = None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input batch of shape (B, 3, H, W).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        # TODO: Pass x through features → pool → flatten → classifier.
        #
        #   The flatten step converts (B, 128, 1, 1) → (B, 128).
        #   Use x.view(x.size(0), -1) to flatten while keeping batch dim.
        pass


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    print("=== VERIFICATION ===")

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


    def _describe(layer):
        """Short, comparable description of a layer, e.g. 'Conv2d(3, 32, kernel_size=3, padding=1)'."""
        if isinstance(layer, nn.Conv2d):
            return (f"Conv2d({layer.in_channels}, {layer.out_channels}, "
                    f"kernel_size={layer.kernel_size[0]}, padding={layer.padding[0]})")
        if isinstance(layer, nn.BatchNorm2d):
            return f"BatchNorm2d({layer.num_features})"
        if isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size
            return f"MaxPool2d({k if isinstance(k, int) else k[0]})"
        if isinstance(layer, nn.Linear):
            return f"Linear({layer.in_features}, {layer.out_features})"
        if isinstance(layer, nn.Dropout):
            return f"Dropout(p={layer.p})"
        return type(layer).__name__


    def _assert_layers(module, attr, expected):
        _assert_true(isinstance(module, nn.Sequential),
                     f"expected self.{attr} to be an nn.Sequential, got {type(module).__name__}")
        got = [_describe(layer) for layer in module]
        for i, (g, e) in enumerate(zip(got, expected)):
            _assert_true(g == e, f"self.{attr} layer {i}: expected {e}, got {g}")
        _assert_true(len(got) == len(expected), f"expected {len(expected)} layers in self.{attr}, got {len(got)}")


    def _conv_block(c_in, c_out):
        return [f"Conv2d({c_in}, {c_out}, kernel_size=3, padding=1)", f"BatchNorm2d({c_out})", "ReLU", "MaxPool2d(2)"]


    EXPECTED_FEATURES = _conv_block(3, 32) + _conv_block(32, 64) + _conv_block(64, 128)
    EXPECTED_CLASSIFIER = ["Linear(128, 64)", "ReLU", "Dropout(p=0.5)", "Linear(64, 10)"]
    EXPECTED_PARAM_COUNT = 102_602  # fixed by the architecture spec above (num_classes=10)


    def _check_output_shape(input_shape, num_classes):
        model = PlantCNN(num_classes=num_classes).eval()
        with torch.no_grad():
            output = model(torch.randn(*input_shape))
        _assert_true(output is not None, "forward() returned None - did you return the logits?")
        expected = (input_shape[0], num_classes)
        _assert_true(tuple(output.shape) == expected, f"expected output shape {expected}, got {tuple(output.shape)}")


    def _check_dropout_arg():
        classifier = PlantCNN(num_classes=10, dropout=0.3).classifier
        _assert_true(isinstance(classifier, nn.Sequential), "self.classifier is not defined yet")
        dropouts = [m for m in classifier if isinstance(m, nn.Dropout)]
        _assert_true(len(dropouts) == 1 and dropouts[0].p == 0.3,
                     "expected PlantCNN(dropout=0.3) to use Dropout(0.3) - pass the dropout argument through")


    results = []

    results.append(check("PlantCNN - self.features is three Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2) blocks (3->32->64->128)",
                         lambda: _assert_layers(PlantCNN(num_classes=10).features, "features", EXPECTED_FEATURES)))
    results.append(check("PlantCNN - self.pool is AdaptiveAvgPool2d(1)",
                         lambda: _assert_true(isinstance(PlantCNN(num_classes=10).pool, nn.AdaptiveAvgPool2d)
                                              and PlantCNN(num_classes=10).pool.output_size in (1, (1, 1)),
                                              "expected self.pool = nn.AdaptiveAvgPool2d(1)")))
    results.append(check("PlantCNN - self.classifier is Linear(128, 64) -> ReLU -> Dropout -> Linear(64, num_classes)",
                         lambda: _assert_layers(PlantCNN(num_classes=10, dropout=0.5).classifier, "classifier",
                                                EXPECTED_CLASSIFIER)))
    results.append(check("PlantCNN - the dropout argument is used in the classifier", _check_dropout_arg))
    results.append(check("PlantCNN - forward maps (4, 3, 128, 128) to (4, 10)",
                         lambda: _check_output_shape((4, 3, 128, 128), num_classes=10)))
    results.append(check("PlantCNN - forward works for other image sizes, (2, 3, 64, 64) -> (2, 10)",
                         lambda: _check_output_shape((2, 3, 64, 64), num_classes=10)))
    results.append(check("PlantCNN - num_classes sets the output size, (2, 3, 128, 128) -> (2, 5)",
                         lambda: _check_output_shape((2, 3, 128, 128), num_classes=5)))
    results.append(check(f"PlantCNN - has {EXPECTED_PARAM_COUNT:,} trainable parameters",
                         lambda: _assert_true(count_parameters(PlantCNN(num_classes=10)) == EXPECTED_PARAM_COUNT,
                                              f"expected {EXPECTED_PARAM_COUNT:,}, got "
                                              f"{count_parameters(PlantCNN(num_classes=10)):,}")))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
