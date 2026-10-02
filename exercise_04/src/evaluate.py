"""
Exercise 03 - Standalone Classification Evaluation (Fully Coded)
===================================================================
Load a saved checkpoint and evaluate on the test (or val) set.
Computes: accuracy, per-class accuracy, precision, recall, F1,
and prints a confusion matrix.
"""

import os
import sys
import argparse

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PlantWildDataset
from transforms import get_val_transforms
from model import PlantCNN, get_model, count_parameters


def evaluate(model, dataloader, device, class_names):
    """
    Evaluate a classification model on a dataset split.

    Args:
        model: Classification model.
        dataloader: DataLoader for the split to evaluate.
        device: torch.device.
        class_names: list of class name strings.

    Returns:
        dict: Contains accuracy, per_class_accuracy, precision, recall,
              f1, confusion_matrix.
    """
    model.eval()
    num_classes = len(class_names)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Overall accuracy
    correct = (all_preds == all_targets).sum().item()
    total = all_targets.size(0)
    accuracy = correct / total

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        cm[t.item(), p.item()] += 1

    # Per-class metrics
    per_class_accuracy = []
    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        class_total = cm[i, :].sum()

        # Per-class accuracy
        per_class_accuracy.append(tp / class_total if class_total > 0 else 0.0)

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision.append(prec)

        # Recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall.append(rec)

        # F1
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1.append(f1_val)

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def print_results(metrics, class_names, split="test"):
    """Pretty-print evaluation results."""
    print(f"\n  [{split.upper()}] Results:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f} "
          f"({int(metrics['accuracy'] * sum(sum(metrics['confusion_matrix'])))}"
          f"/{sum(sum(metrics['confusion_matrix']))})")

    print(f"\n  Per-class results:")
    print(f"  {'Class':<25} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Count':>7}")
    print(f"  {'-' * 62}")
    for i, name in enumerate(class_names):
        count = metrics["confusion_matrix"][i, :].sum()
        print(f"  {name:<25} "
              f"{metrics['per_class_accuracy'][i]:>7.4f} "
              f"{metrics['precision'][i]:>7.4f} "
              f"{metrics['recall'][i]:>7.4f} "
              f"{metrics['f1'][i]:>7.4f} "
              f"{count:>7}")

    # Macro averages
    avg_prec = np.mean(metrics["precision"])
    avg_rec = np.mean(metrics["recall"])
    avg_f1 = np.mean(metrics["f1"])
    print(f"  {'-' * 62}")
    print(f"  {'Macro Average':<25} {'':>7} {avg_prec:>7.4f} {avg_rec:>7.4f} {avg_f1:>7.4f}")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    num_classes = len(class_names)
    max_name_len = max(len(n) for n in class_names)
    col_width = max(max_name_len, 6)

    print(f"\n  Confusion Matrix (rows = true, columns = predicted):")
    header = "  " + " " * (max_name_len + 2)
    for name in class_names:
        header += f"{name[:col_width]:>{col_width}} "
    print(header)

    for i, name in enumerate(class_names):
        row = f"  {name:<{max_name_len}}  "
        for j in range(num_classes):
            row += f"{cm[i, j]:>{col_width}} "
        print(row)


# ===========================================================================
# Standalone evaluation
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--model", type=str, default=None,
                        help="Architecture of the checkpoint. Only needed for old checkpoints "
                             "that don't store it (default: model.name from config)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run training first: python src/train.py --config config.yaml")
        sys.exit(1)

    config = yaml.safe_load(open(args.config))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_root = config["data"]["root_dir"]
    image_size = config["data"]["image_size"]

    dataset = PlantWildDataset(
        root_dir=os.path.join(data_root, args.split),
        transform=get_val_transforms(image_size),
    )
    dataloader = DataLoader(
        dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    # Architecture: stored in the checkpoint > --model > config
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name") or args.model or config["model"]["name"]
    if args.model and ckpt.get("model_name") and args.model != ckpt["model_name"]:
        print(f"Warning: --model {args.model} ignored, checkpoint was trained "
              f"with {ckpt['model_name']}")
    print(f"Model architecture: {model_name}")

    model = get_model(
        model_name,
        num_classes=len(dataset.classes),
        pretrained=False,  # weights come from the checkpoint
        dropout=config["model"]["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_accuracy={ckpt['val_accuracy']:.4f})")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Evaluating on {args.split} set ({len(dataset)} images)...")

    metrics = evaluate(model, dataloader, device, dataset.classes)
    print_results(metrics, dataset.classes, split=args.split)
