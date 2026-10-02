"""
Exercise 05 - Detection Model Factory
========================================
Load a pretrained detection model and replace its classification head
for the PhenoBench task (3 classes).

All supported models share the same API:
    Training:   model(images, targets) → loss_dict
    Inference:  model(images) → [{"boxes", "labels", "scores"}, ...]

Swap models by changing model.name in config.yaml.

CNN models (torchvision):
    "fasterrcnn_resnet50", "fasterrcnn_mobilenet", "retinanet_resnet50"

Transformer models (HuggingFace):
    "rtdetr" — Real-Time DETR. Set prediction with a transformer
    encoder-decoder. No anchors, no NMS, no proposals.

Run this script to verify:
    python src/model.py
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from transformers import RTDetrForObjectDetection, RTDetrConfig


# ---------------------------------------------------------------------------
# RT-DETR wrapper (given) — adapts HuggingFace API to torchvision convention
# ---------------------------------------------------------------------------
class RTDetrWrapper(nn.Module):
    """
    Wraps HuggingFace RT-DETR so it behaves like a torchvision detection model.
    Handles format conversion internally:
        - Torchvision: boxes [x1, y1, x2, y2] pixels, key "labels"
        - HuggingFace: boxes [cx, cy, w, h] normalized, key "class_labels"

    Given fully implemented — no TODOs. Read through to understand the
    conversion logic if you are interested.
    """

    def __init__(self, model, score_threshold=0.5):
        super().__init__()
        self.model = model
        self.score_threshold = score_threshold

    def _convert_targets_to_hf(self, targets, image_sizes):
        """Convert torchvision target format → HuggingFace format."""
        hf_targets = []
        for target, (h, w) in zip(targets, image_sizes):
            boxes = target["boxes"].clone().float()

            # [x1, y1, x2, y2] → [cx, cy, w, h]
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]

            # Normalize to [0, 1]
            cx /= w
            cy /= h
            bw /= w
            bh /= h

            hf_targets.append({
                "class_labels": target["labels"] - 1,  # torchvision 1-indexed → DETR 0-indexed
                "boxes": torch.stack([cx, cy, bw, bh], dim=1),
            })
        return hf_targets

    def _convert_preds_to_torchvision(self, outputs, image_sizes):
        """Convert HuggingFace predictions → torchvision format."""
        logits = outputs.logits          # (B, num_queries, num_classes+1)
        pred_boxes = outputs.pred_boxes  # (B, num_queries, 4) [cx,cy,w,h] normalized

        results = []
        for i, (h, w) in enumerate(image_sizes):
            probs = logits[i].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)

            keep = scores > self.score_threshold
            scores = scores[keep]
            labels = labels[keep] + 1  # DETR 0-indexed → torchvision 1-indexed

            boxes_cxcywh = pred_boxes[i][keep]

            # [cx, cy, w, h] normalized → [x1, y1, x2, y2] pixels
            cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0)

            results.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
            })
        return results

    def forward(self, images, targets=None):
        pixel_values = torch.stack(images)
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]

        if self.training and targets is not None:
            hf_targets = self._convert_targets_to_hf(targets, image_sizes)
            outputs = self.model(pixel_values=pixel_values, labels=hf_targets)
            return {"loss_total": outputs.loss}
        else:
            outputs = self.model(pixel_values=pixel_values)
            return self._convert_preds_to_torchvision(outputs, image_sizes)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_detection_model(name, num_classes, pretrained=True):
    """
    Create a detection model and replace its head for num_classes.

    Args:
        name (str): Model identifier from config.yaml.
        num_classes (int): Classes including background (3 for PhenoBench).
        pretrained (bool): Load COCO-pretrained weights.

    Returns:
        nn.Module: Detection model ready for fine-tuning.
    """
    name = name.lower()

    # TODO: Load the requested model and replace its classification head.
    #
    #   Each model type has a different way to swap the head. Handle
    #   each name with an if/elif chain and raise ValueError for unknown names.
    #   Hint: See the imported classes at the top of this file for the right loader functions
    #
    #   --- Faster R-CNN variants ("fasterrcnn_resnet50", "fasterrcnn_mobilenet") ---
    #
    #   Step 1: Choose the right loader function and weights enum based on name.
    #           Set weights to the .DEFAULT member if pretrained, else None.
    #           Call the loader to create the model.
    #
    #   Step 2: The pretrained head classifies 91 COCO classes. To replace it:
    #           - Read how many input features the current head expects from
    #             model.roi_heads.box_predictor.cls_score.in_features
    #           - Create a new FastRCNNPredictor(in_features, num_classes)
    #           - Assign it to model.roi_heads.box_predictor
    #
    if name == "fasterrcnn_resnet50":
        pass

    elif name == "fasterrcnn_mobilenet":
        pass

    #   --- RetinaNet ("retinanet_resnet50") ---
    #
    #   Step 1: Load with retinanet_resnet50_fpn(weights=...).
    #
    #   Step 2: The classification head lives at model.head.classification_head.
    #           Read in_channels from its .cls_logits.in_channels and num_anchors
    #           from its .num_anchors attribute.
    #           Replace it with a new RetinaNetClassificationHead(
    #               in_channels, num_anchors, num_classes).
    #
    elif name == "retinanet_resnet50":
        pass
    

    #   --- RT-DETR ("rtdetr") — Transformer detector ---
    elif name == "rtdetr":
        from transformers import RTDetrForObjectDetection, RTDetrConfig

        # DETR handles background internally — subtract 1
        detr_num_classes = num_classes - 1

        if pretrained:
            hf_model = RTDetrForObjectDetection.from_pretrained(
                "PekingU/rtdetr_r50vd"
            )
        else:
            config = RTDetrConfig()
            hf_model = RTDetrForObjectDetection(config)

        in_features = hf_model.class_embed[0].in_features
        hf_model.class_embed = nn.ModuleList([
            nn.Linear(in_features, detr_num_classes)
            for _ in range(len(hf_model.class_embed))
        ])

        model = RTDetrWrapper(hf_model)

    else:
        raise ValueError(
            f"Unknown model: '{name}'. Choose from: "
            f"fasterrcnn_resnet50, fasterrcnn_mobilenet, "
            f"retinanet_resnet50, rtdetr"
        )

    return model

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    num_classes = 3
    model_names = [
        "fasterrcnn_resnet50",
        "fasterrcnn_mobilenet",
        "retinanet_resnet50",
        "rtdetr",
    ]

    print(f"{'Model':<25} {'Params':>12} {'Type':<20}")
    print("-" * 60)

    for name in model_names:
        model = get_detection_model(name, num_classes=num_classes, pretrained=False)
        params = count_parameters(model)
        arch_type = "transformer" if "detr" in name else "CNN"
        print(f"{name:<25} {params:>12,} {arch_type:<20}")

    print("\nVerifying training API compatibility...")
    dummy_images = [torch.randn(3, 256, 256)]
    dummy_targets = [{
        "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]

    for name in model_names:
        model = get_detection_model(name, num_classes=num_classes, pretrained=False)
        model.train()
        loss_dict = model(dummy_images, dummy_targets)
        total_loss = sum(loss_dict.values())

        model.eval()
        with torch.no_grad():
            preds = model(dummy_images)

        print(f"  {name:<25} train_loss={total_loss.item():.3f}  "
              f"pred_boxes={preds[0]['boxes'].shape}")

    print("\nAll models (CNN + transformer) share the same API.")
    print("\n--- Model verified. ---")
