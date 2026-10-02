"""
Optional — MixUp Data Augmentation
=====================================
MixUp creates virtual training samples by linearly interpolating
between random pairs of images and their labels. This acts as a
regularizer and often improves generalization.

Given two samples (x_i, y_i) and (x_j, y_j) and a mixing coefficient
lambda ~ Beta(alpha, alpha):
    x_mixed = lambda * x_i + (1 - lambda) * x_j
    y_mixed = lambda * y_i + (1 - lambda) * y_j

Since labels become soft (no longer one-hot integers), the loss function
must handle soft targets.

This is a complete script for instructor demonstration.

Usage:
    python optional/mixup.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN
from utils import set_seed


def mixup_data(x, y, alpha=0.2):
    """
    Apply MixUp to a batch.

    Args:
        x (Tensor): Input images, shape (B, C, H, W).
        y (Tensor): Labels, shape (B,).
        alpha (float): Beta distribution parameter. Higher = more mixing.

    Returns:
        mixed_x: Interpolated images.
        y_a, y_b: Original label pairs.
        lam: Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    """
    Compute mixed loss for MixUp.

    Args:
        criterion: Loss function (e.g. CrossEntropyLoss).
        outputs: Model predictions.
        y_a, y_b: Label pairs from mixup_data.
        lam: Mixing coefficient.

    Returns:
        Mixed loss value.
    """
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def main(config_path, alpha=0.2):
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

    model = PlantCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_epochs = min(config["training"]["epochs"], 10)
    print(f"Training with MixUp (alpha={alpha}) for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        # Train with MixUp
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Apply MixUp
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=alpha)

            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validate (no MixUp during validation)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_dataset)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    main(args.config, alpha=args.alpha)
