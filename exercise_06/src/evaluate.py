"""
Exercise 06 - Segmentation Evaluation
========================================
Compute mIoU, per-class IoU, Dice coefficient, and pixel accuracy.
This file serves two roles:

1. Imported by train.py — called after each epoch on the val set.
2. Run standalone — evaluate a saved checkpoint on the test (or val) set.

Standalone usage:
    python src/evaluate.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_unet_best.pt

    python src/evaluate.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_unet_best.pt \
        --split val
"""

import os
import sys
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchSegDataset
from transforms import get_val_transforms
from model import get_segmentation_model
from utils import load_checkpoint


CLASS_NAMES = ["background", "crop", "weed"]


def compute_confusion_matrix(preds, targets, num_classes):
    """
    Build a confusion matrix from predictions and targets.

    Args:
        preds (Tensor): Predicted class indices, shape (B, H, W).
        targets (Tensor): Ground truth class indices, shape (B, H, W).
        num_classes (int): Number of classes.

    Returns:
        ndarray: Shape (num_classes, num_classes). Row = true, Col = predicted.
    """
    # Build the confusion matrix.
    #
    #   Step 1: Flatten both preds and targets to 1D.
    #
    #   Step 2: Filter out pixels where the target is outside [0, num_classes).
    #           Hint: this is the same as keeping only valid pixels for evaluation. Use a boolean mask.
    #
    #   Step 3: Compute a linear index: target * num_classes + pred.
    #           Each (true, predicted) pair maps to a unique position.
    #
    #   Step 4: Use torch.bincount on the index with
    #           minlength=num_classes*num_classes, then reshape to
    #           (num_classes, num_classes). Convert to numpy.

    preds = preds.flatten()
    targets = targets.flatten()

    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    index = targets * num_classes + preds
    cm = torch.bincount(index, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).cpu().numpy()
    


def compute_metrics(confusion_matrix):
    """
    Compute segmentation metrics from a confusion matrix.

    Args:
        confusion_matrix (ndarray): Shape (num_classes, num_classes).

    Returns:
        dict: pixel_accuracy, per_class_iou, miou, per_class_dice, mean_dice.
    """
    # Compute all metrics from the confusion matrix.
    #
    #   Step 1: Pixel accuracy = sum of diagonal / sum of entire matrix.
    #           hint: np.diag() and .sum() will be useful.
    #
    #   Step 2: Per-class IoU. For each class i:
    #           TP = diagonal element [i, i]
    #           FP = column i sum minus TP
    #           FN = row i sum minus TP
    #           IoU = TP / (TP + FP + FN)
    #           Handle TP + FP + FN == 0 (class absent).
    #
    #   Step 3: mIoU = mean of per-class IoUs (only classes present).
    #
    #   Step 4: Per-class Dice = 2*TP / (2*TP + FP + FN).
    #           Same edge case handling as IoU.
    #
    #   Step 5: Mean Dice = mean of per-class Dice values.

    pixel_accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    
    per_class_iou = []
    per_class_dice = []

    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        denom = tp + fp + fn

        if denom == 0:
            per_class_iou.append(0.0)
            per_class_dice.append(0.0)
        else:
            per_class_iou.append(tp / denom)
            per_class_dice.append(2 * tp / (2 * tp + fp + fn))

    valid_iou = [v for v in per_class_iou if v > 0]
    valid_dice = [v for v in per_class_dice if v > 0]

    miou = np.mean(valid_iou) if valid_iou else 0.0
    mean_dice = np.mean(valid_dice) if valid_dice else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": per_class_iou,
        "miou": miou,
        "per_class_dice": per_class_dice,
        "mean_dice": mean_dice,
    }


def evaluate(model, dataloader, device, num_classes):
    """
    Run evaluation on a dataset split.

    Called by train.py (val set) and standalone mode (test or val set).
    Does not know or care which split — just evaluates whatever
    dataloader it receives.

    Args:
        model: Segmentation model.
        dataloader: DataLoader for the split to evaluate.
        device: torch.device.
        num_classes (int): Number of classes.

    Returns:
        dict: Metric results from compute_metrics().
    """
    model.eval()
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Accumulate the confusion matrix over all batches.
    #
    #   Step 1: Loop over the dataloader inside torch.no_grad().
    #
    #   Step 2: Move images AND masks to device (preds and masks must be
    #           on the same device for step 4). Forward pass through the model.
    #           If model output is a dict (torchvision models), extract
    #           the "out" key. U-Net and SegFormer wrapper return a
    #           plain tensor.
    #
    #   Step 3: Get predicted classes: output.argmax(dim=1).
    #
    #   Step 4: Compute the confusion matrix for this batch using
    #           compute_confusion_matrix() and add it to
    #           the running total total_cm 

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            output = model(images)
            if isinstance(output, dict):
                output = output["out"]

            preds = output.argmax(dim=1)
            total_cm += compute_confusion_matrix(preds, masks, num_classes)


    

    return compute_metrics(total_cm)


def print_results(metrics, split="val"):
    """Pretty-print evaluation results."""
    print(f"\n  [{split.upper()}] Results:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  mIoU:           {metrics['miou']:.4f}")
    print(f"  Mean Dice:      {metrics['mean_dice']:.4f}")

    print(f"\n  Per-class results:")
    print(f"  {'Class':<15} {'IoU':>8} {'Dice':>8}")
    print(f"  {'-'*33}")
    for i, name in enumerate(CLASS_NAMES):
        iou = metrics["per_class_iou"][i]
        dice = metrics["per_class_dice"][i]
        print(f"  {name:<15} {iou:>8.4f} {dice:>8.4f}")


# ===========================================================================
# Standalone evaluation
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run training first: python src/train.py --config config.yaml")
        sys.exit(1)

    config = yaml.safe_load(open(args.config))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_root = config["data"]["root_dir"]

    dataset = PhenoBenchSegDataset(
        root_dir=os.path.join(data_root, args.split),
        transforms=get_val_transforms(config["data"]["image_size"]),
    )
    dataloader = DataLoader(
        dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    model = get_segmentation_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Evaluating on {args.split} set ({len(dataset)} images)...")

    metrics = evaluate(model, dataloader, device, config["model"]["num_classes"])
    print_results(metrics, split=args.split)
