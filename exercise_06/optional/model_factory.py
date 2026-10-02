"""
Optional — Segmentation Model Factory
=========================================
Swap PhenoSegNet for pretrained SOTA models from torchvision.
All models share the same API: input (B, 3, H, W) → output (B, num_classes, H, W).

To use: replace PhenoSegNet in train.py with get_segmentation_model(),
or add a model.name field to config.yaml and an if/else in train.py.

Note: torchvision segmentation models return a dict ({"out": tensor, ...})
instead of a plain tensor. Handle this in the training/eval loop:
    output = model(images)
    if isinstance(output, dict):
        output = output["out"]

Usage:
    python optional/model_factory.py
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import PhenoSegNet, count_parameters


def get_segmentation_model(name, num_classes, pretrained=True, base_channels=32):
    """
    Factory function for segmentation models.

    Args:
        name (str): "phenosegnet", "deeplabv3_resnet50", or "fcn_resnet50".
        num_classes (int): Number of output classes (3 for PhenoBench).
        pretrained (bool): Load pretrained weights (ignored for phenosegnet).
        base_channels (int): Base channel count for PhenoSegNet.

    Returns:
        nn.Module: Segmentation model.
    """
    name = name.lower()

    if name == "phenosegnet":
        model = PhenoSegNet(num_classes=num_classes, base_channels=base_channels)

    elif name == "deeplabv3_resnet50":
        # DeepLabV3 with ResNet-50 + atrous spatial pyramid pooling (ASPP).
        # Pretrained on COCO (21 classes). Replace the classifier head
        # and the auxiliary classifier head for num_classes.
        weights = "DEFAULT" if pretrained else None
        model = seg_models.deeplabv3_resnet50(weights=weights)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    elif name == "fcn_resnet50":
        # Fully Convolutional Network with ResNet-50 backbone.
        # Simpler than DeepLabV3, also pretrained on COCO.
        weights = "DEFAULT" if pretrained else None
        model = seg_models.fcn_resnet50(weights=weights)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    else:
        raise ValueError(
            f"Unknown model: '{name}'. Choose from: "
            f"phenosegnet, deeplabv3_resnet50, fcn_resnet50"
        )

    return model


if __name__ == "__main__":
    num_classes = 3
    dummy_input = torch.randn(2, 3, 256, 256)

    model_names = ["phenosegnet", "deeplabv3_resnet50", "fcn_resnet50"]

    print(f"{'Model':<22} {'Params':>12} {'Output Shape':>18}")
    print("-" * 55)

    for name in model_names:
        model = get_segmentation_model(name, num_classes=num_classes, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                output = output["out"]

        params = count_parameters(model)
        print(f"{name:<22} {params:>12,} {str(list(output.shape)):>18}")

    print("\nAll models output (B, num_classes, H, W) matching input spatial dims.")
    print("\nIMPORTANT: torchvision models return a dict, not a tensor.")
    print("In train.py, add after forward pass:")
    print("    if isinstance(output, dict): output = output['out']")
    print("\n--- Model factory verified. ---")
