"""
Optional — Weighted Cross-Entropy Loss
========================================
When classes are imbalanced (some have many more images than others),
standard CrossEntropyLoss treats all samples equally, which biases
the model toward the majority class. Weighted CE assigns higher loss
to under-represented classes.

This script computes class weights from the training set and uses them
in nn.CrossEntropyLoss(weight=...).

This is a complete script for instructor demonstration.

Usage:
    python optional/weighted_loss.py --config config.yaml
"""

import os
import sys
import argparse
from collections import Counter

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN
from utils import set_seed


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights.

    Classes with fewer samples get higher weight, so the loss penalizes
    misclassifying rare classes more heavily.

    Args:
        dataset: PlantWildDataset instance.

    Returns:
        Tensor: Weight per class, shape (num_classes,).
    """
    label_counts = Counter([label for _, label in dataset.samples])
    num_classes = len(dataset.classes)
    total_samples = len(dataset)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        # Inverse frequency: total / (num_classes * count_for_this_class)
        w = total_samples / (num_classes * count)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()

    return weights


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    image_size = config["data"]["image_size"]

    train_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "train"),
        transform=get_train_transforms(image_size, augmentation=True),
    )
    val_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "val"),
        transform=get_val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    # Compute and display class weights
    class_weights = compute_class_weights(train_dataset)

    print("Class weights (inverse frequency, normalized):")
    for i, cls_name in enumerate(train_dataset.classes):
        count = Counter([l for _, l in train_dataset.samples])[i]
        print(f"  {cls_name}: weight={class_weights[i]:.3f} (n={count})")

    # Standard loss vs. weighted loss
    criterion_standard = nn.CrossEntropyLoss()
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = PlantCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Compare loss values on one batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        loss_std = criterion_standard(outputs, labels)
        loss_wt = criterion_weighted(outputs, labels)

    print(f"\nSample batch comparison:")
    print(f"  Standard CE loss: {loss_std.item():.4f}")
    print(f"  Weighted CE loss: {loss_wt.item():.4f}")
    print(f"\nTo use weighted loss in training, replace:")
    print(f"  criterion = nn.CrossEntropyLoss()")
    print(f"with:")
    print(f"  weights = compute_class_weights(train_dataset).to(device)")
    print(f"  criterion = nn.CrossEntropyLoss(weight=weights)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
