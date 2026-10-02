"""
Optional — Faster R-CNN with Different Backbones
===================================================
Demonstrates swapping the Faster R-CNN backbone from ResNet-50 to
lighter alternatives (ResNet-18 via backbone_utils, MobileNetV3).

Usage:
    python optional/faster_rcnn_backbones.py
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_mobilenet_fasterrcnn(num_classes, pretrained=True):
    """
    Faster R-CNN with MobileNetV3-Large backbone.
    Much lighter than ResNet-50 — faster training, lower accuracy.
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    num_classes = 3
    dummy_input = [torch.randn(3, 800, 800)]

    # Compare backbones
    models = {
        "ResNet-50 FPN": None,  # from main model.py
        "MobileNetV3 FPN": get_mobilenet_fasterrcnn(num_classes, pretrained=False),
    }

    # Add ResNet-50 for comparison
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    resnet_model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = resnet_model.roi_heads.box_predictor.cls_score.in_features
    resnet_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    models["ResNet-50 FPN"] = resnet_model

    print(f"{'Backbone':<22} {'Params':>12}")
    print("-" * 36)

    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:<22} {params:>12,}")

    print("\nMobileNetV3 is ~3x smaller than ResNet-50.")
    print("Trade-off: faster training but lower detection accuracy.")
    print("To use: change get_detection_model() in model.py or add a")
    print("'backbone' field in config.yaml.")
