"""
Exercise 05 - Detection Evaluation (Fully Coded)
====================================================
COCO evaluation using pycocotools. This file serves two roles:

1. Imported by train.py — called after each epoch on the val set.
2. Run standalone — evaluate a saved checkpoint on the test (or val) set.

Standalone usage:
    python src/evaluate.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_fasterrcnn_best.pt

    python src/evaluate.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_fasterrcnn_best.pt \
        --split val
"""

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchDetectionDataset
from transforms import get_val_transforms
from model import get_detection_model
from utils import collate_fn, load_checkpoint


def evaluate(model, dataloader, device, annotation_file):
    """
    Evaluate a detection model using COCO metrics.

    Called by train.py (val set) and standalone mode (test or val set).
    Does not know or care which split — just evaluates whatever
    dataloader it receives.

    Args:
        model: Detection model.
        dataloader: DataLoader for the split to evaluate.
        device: torch.device.
        annotation_file (str): Path to COCO JSON for this split.

    Returns:
        dict: Contains "coco_stats" (list of 12 COCO metrics),
              "ap50_95", "ap50", "ap75", "ar100".
    """
    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                image_id = target["image_id"].item()

                boxes = pred["boxes"].cpu()
                labels = pred["labels"].cpu()
                scores = pred["scores"].cpu()

                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.tolist()
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score.item(),
                    })

    if len(coco_results) == 0:
        print("  No predictions — returning zero metrics.")
        return {
            "coco_stats": [0.0] * 12,
            "ap50_95": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
            "ar100": 0.0,
        }

    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist()

    return {
        "coco_stats": stats,
        "ap50_95": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "ar100": stats[8],
    }


def print_results(metrics, split="val"):
    """Pretty-print evaluation results."""
    print(f"\n  [{split.upper()}] Results:")
    print(f"  AP@[.50:.95]: {metrics['ap50_95']:.4f}")
    print(f"  AP@.50:       {metrics['ap50']:.4f}")
    print(f"  AP@.75:       {metrics['ap75']:.4f}")
    print(f"  AR@100:       {metrics['ar100']:.4f}")


# ===========================================================================
# Standalone evaluation
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection model")
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

    dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, args.split),
        annotation_file=os.path.join(data_root, "annotations", f"{args.split}.json"),
        transforms=get_val_transforms(),
    )
    dataloader = DataLoader(
        dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = get_detection_model(
    name=config["model"]["name"],
    num_classes=config["model"]["num_classes"],
    pretrained=(config["model"]["name"].lower() == "rtdetr"),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Evaluating on {args.split} set ({len(dataset)} images)...")

    metrics = evaluate(
        model, dataloader, device,
        annotation_file=os.path.join(data_root, "annotations", f"{args.split}.json"),
    )
    print_results(metrics, split=args.split)
