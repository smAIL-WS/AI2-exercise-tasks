"""
Exercise 07 - Final Comparison: Few-Shot SegFormer vs. Exercise 06's U-Net
==============================================================================
Fully coded -- the interesting work already happened in model.py, train.py,
and evaluate.py. This script just runs both checkpoints against the shared
PhenoBench test set and prints them side by side.

Loads Exercise 06's model.py directly from disk (via importlib, under a
distinct module name) rather than duplicating the U-Net architecture here
-- if you haven't completed Exercise 06, this script cannot rebuild that
model. It never imports or modifies anything else from Exercise 06.

Usage:
    python src/compare_to_ex06.py \
        --config config.yaml \
        --segformer_checkpoint outputs/checkpoints/phenobench_segformer_fewshot_best.pt \
        --ex06_src ../exercise_06/src \
        --ex06_config ../exercise_06/config.yaml \
        --ex06_checkpoint ../exercise_06/outputs/checkpoints/phenobench_unet_best.pt
"""

import os
import sys
import time
import argparse
import importlib.util

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PhenoBenchFewShotSegDataset
from transforms import get_val_transforms
from model import load_pretrained_segformer, parameter_summary
from evaluate import compute_confusion_matrix, compute_metrics
from utils import load_checkpoint

CLASS_NAMES = ["background", "crop", "weed"]


def load_ex06_model_module(ex06_src_dir):
    """
    Load exercise_06/src/model.py under the distinct module name
    "ex06_model" so it doesn't collide with this exercise's own model.py
    (Python caches imported modules by name, and both files are literally
    named "model.py").
    """
    model_path = os.path.join(ex06_src_dir, "model.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {model_path}. Pass --ex06_src pointing at "
            f"your Exercise 06 checkout's src/ folder."
        )
    spec = importlib.util.spec_from_file_location("ex06_model", model_path)
    ex06_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex06_model)
    return ex06_model


def evaluate_segformer(model, dataloader, device, num_classes):
    model.eval()
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            total_cm += compute_confusion_matrix(preds, masks, num_classes)
    return compute_metrics(total_cm)


def evaluate_unet(model, dataloader, device, num_classes):
    """Exercise 06's models return a plain tensor (UNet) or a dict with an
    "out" key (torchvision's DeepLabV3/FCN) -- handle both, exactly like
    Exercise 06's own evaluate.py does."""
    model.eval()
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            if isinstance(output, dict):
                output = output["out"]
            preds = output.argmax(dim=1)
            total_cm += compute_confusion_matrix(preds, masks, num_classes)
    return compute_metrics(total_cm)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                         help="This exercise's config.yaml")
    parser.add_argument("--segformer_checkpoint", type=str, required=True)
    parser.add_argument("--ex06_src", type=str, default="../exercise_06/src",
                         help="Path to your Exercise 06 checkout's src/ folder")
    parser.add_argument("--ex06_config", type=str, default="../exercise_06/config.yaml")
    parser.add_argument("--ex06_checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    num_classes = config["model"]["num_classes"]

    test_dataset = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "test"),
        transforms=get_val_transforms(config["data"]["image_size"]),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )
    print(f"Test images: {len(test_dataset)}\n")

    # --- Exercise 07: few-shot SegFormer ---
    segformer = load_pretrained_segformer(config["model"]["name"], num_classes).to(device)
    load_checkpoint(args.segformer_checkpoint, segformer)
    seg_summary = parameter_summary(segformer)

    start = time.time()
    segformer_metrics = evaluate_segformer(segformer, test_loader, device, num_classes)
    segformer_time = time.time() - start

    # --- Exercise 06: full-data U-Net ---
    ex06_model_module = load_ex06_model_module(args.ex06_src)
    ex06_config = yaml.safe_load(open(args.ex06_config))
    unet = ex06_model_module.get_segmentation_model(
        ex06_config["model"]["name"], num_classes, pretrained=False,
    ).to(device)
    load_checkpoint(args.ex06_checkpoint, unet)

    start = time.time()
    unet_metrics = evaluate_unet(unet, test_loader, device, num_classes)
    unet_time = time.time() - start

    # --- Report ---
    print("=" * 72)
    print(f"{'Metric':<24} {'Ex06 U-Net (full data)':>22} {'Ex07 SegFormer (few-shot)':>24}")
    print("-" * 72)
    print(f"{'mIoU':<24} {unet_metrics['miou']:>22.4f} {segformer_metrics['miou']:>24.4f}")
    print(f"{'Pixel Accuracy':<24} {unet_metrics['pixel_accuracy']:>22.4f} {segformer_metrics['pixel_accuracy']:>24.4f}")
    print(f"{'Trainable params':<24} {count_parameters(unet):>22,} {seg_summary['trainable']:>24,}")
    print(f"{'Training images used':<24} {'full train split':>22} {config['data']['num_shots']:>24}")
    print(f"{'Eval wall-time (s)':<24} {unet_time:>22.2f} {segformer_time:>24.2f}")
    print("=" * 72)

    print("\nPer-class IoU:")
    print(f"  {'Class':<15} {'Ex06 U-Net':>12} {'Ex07 SegFormer':>16}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15} {unet_metrics['per_class_iou'][i]:>12.4f} {segformer_metrics['per_class_iou'][i]:>16.4f}")


if __name__ == "__main__":
    main()
