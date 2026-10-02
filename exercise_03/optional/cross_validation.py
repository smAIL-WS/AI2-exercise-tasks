"""
Optional — K-Fold Cross-Validation
====================================
Demonstrates how to set up k-fold cross-validation instead of a fixed
train/val split. Uses sklearn's KFold to create fold indices, then
trains and evaluates a separate model per fold.

This is a complete script for instructor demonstration.

Usage:
    python optional/cross_validation.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN
from utils import set_seed


def main(config_path, k_folds=5):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    image_size = config["data"]["image_size"]

    # Load the full training set (we split it ourselves into k folds)
    full_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "train"),
        transform=None,  # transforms applied per-fold below
    )

    train_transforms = get_train_transforms(image_size, augmentation=True)
    val_transforms = get_val_transforms(image_size)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config["seed"])
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        print(f"Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")

        # Create subsets with appropriate transforms
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Apply transforms via a wrapper
        train_subset.dataset.transform = train_transforms
        val_subset.dataset.transform = val_transforms

        train_loader = DataLoader(
            train_subset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )

        # Fresh model per fold
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

        # Train for fewer epochs per fold to keep runtime reasonable
        num_epochs = min(config["training"]["epochs"], 10)
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Train
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)

            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"  Epoch {epoch}/{num_epochs} — Val Acc: {val_acc:.4f}")

        fold_results.append(best_val_acc)
        print(f"  Fold {fold + 1} best val accuracy: {best_val_acc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({k_folds} folds)")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i+1}: {acc:.4f}")
    print(f"  Mean:  {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.config, k_folds=args.folds)
