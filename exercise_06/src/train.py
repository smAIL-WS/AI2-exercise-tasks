"""
Exercise 06 - Segmentation Training Script
=============================================
Closer to classification (Ex03) than detection (Ex05) — explicit loss
function, model outputs a tensor not a loss dict.

Usage:
    python src/train.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchSegDataset
from transforms import get_train_transforms, get_val_transforms
from model import get_segmentation_model, count_parameters
from evaluate import evaluate, print_results
from utils import set_seed, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    """
    Train for one epoch. Returns average loss.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        # TODO: Complete one training step.
        #
        #   Step 1: Move images and masks to device.
        #
        #   Step 2: Forward pass. The model returns (B, num_classes, H, W).
        #           If the model is a torchvision model (DeepLabV3, FCN),
        #           the output is a dict — extract the "out" key.
        #           Check with isinstance(output, dict).
        #
        #   Step 3: Compute loss using the criterion. CrossEntropyLoss
        #           accepts (B, num_classes, H, W) predictions directly
        #           against (B, H, W) integer masks.
        #
        #   Step 4: Zero gradients.
        #
        #   Step 5: Backward and optimizer step.


        # --- Logging (do not modify) ---
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    return running_loss / total_samples


def main(config_path):
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
    writer = SummaryWriter(log_dir=log_dir)

    data_root = config["data"]["root_dir"]
    image_size = config["data"]["image_size"]

    # TODO: Create training and validation datasets.
    #   Use PhenoBenchSegDataset with root_dir pointing to "train" or "val"
    #   subfolder, and the appropriate transforms.
    train_dataset = None
    val_dataset = None

    # TODO: Create DataLoaders.
    #   No custom collate_fn needed — default stacking works.
    #   Use batch_size and num_workers from config.
    train_loader = None
    val_loader = None

    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # TODO: Create the model using get_segmentation_model() with name,
    #   num_classes, and pretrained from config. Move to device.
    model = None

    print(f"Model: {config['model']['name']} | Params: {count_parameters(model):,}")

    # TODO: Define loss function — nn.CrossEntropyLoss().
    criterion = None

    # TODO: Define optimizer — AdamW with lr and weight_decay from config.
    optimizer = None

    # --- Training Loop ---
    best_miou = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )
        print(f"  Train Loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, device, config["model"]["num_classes"])
        current_miou = metrics["miou"]

        print(f"  Val mIoU:    {current_miou:.4f}")
        print(f"  Val Px Acc:  {metrics['pixel_accuracy']:.4f}")

        # TODO: Log metrics to TensorBoard using writer.add_scalar():
        #   "Loss/train", "mIoU/val", "PixelAcc/val"
        #   Use epoch as global_step.


        # TODO: Save checkpoint if mIoU improved over best_miou.
        #   Use save_checkpoint() from utils.


    writer.close()
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
