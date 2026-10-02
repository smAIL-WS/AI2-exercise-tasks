"""
Exercise 03 - Training Script
================================
Main training script that ties together dataset, model, loss, optimizer,
TensorBoard logging, and checkpointing.

Usage:
    python src/train.py --config config.yaml

Project structure inspired by: https://github.com/victoresque/pytorch-template
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

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN, count_parameters
from utils import set_seed, compute_accuracy, save_checkpoint


def load_config(config_path):
    """Load YAML config file and return as dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    """
    Train the model for one epoch.

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    # TODO: Set the model to training mode (.train).
    #       This enables dropout and batch normalization updates.


    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # TODO: Complete the training step.
        #
        #   Step 1: Move images and labels to the device using .to(device).
        #
        #   Step 2: Clear old gradients from the optimizer.
        #
        #   Step 3: Forward pass — get model outputs from the images.
        #
        #   Step 4: Compute the loss using the criterion.
        #           Pass (outputs, labels) — not (labels, outputs).
        #
        #   Step 5: Backward pass — compute gradients from the loss.
        #
        #   Step 6: Update weights with the optimizer.


        # --- Accumulate metrics (do not modify) ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.

    Returns:
        tuple: (average_loss, average_accuracy) for the validation set.
    """
    # TODO: Set the model to evaluation mode (.eval).
    #       This disables dropout and freezes batch normalization.


    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # TODO: Disable gradient computation for the entire validation loop.
    #       Wrap the loop in a torch.no_grad() context manager.
    #       Inside, for each batch:
    #
    #   Step 1: Move images and labels to device.
    #   Step 2: Forward pass only — no backward, no optimizer step.
    #   Step 3: Compute loss (for logging, not for training).
    #
    #   The metric accumulation code below handles the rest.

    for images, labels in dataloader:

        pass  # Replace with your solution

        # --- Accumulate metrics (do not modify) ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main(config_path):
    """Main training function."""

    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Create the TensorBoard SummaryWriter. (Hint: https://docs.pytorch.org/docs/2.14/tensorboard.html)
    #   The log directory should include the experiment name so each
    #   experiment gets its own subfolder. Combine config["logging"]["log_dir"]
    #   and config["experiment_name"] using os.path.join().
    log_dir = None
    writer = SummaryWriter(log_dir=log_dir)

    image_size = config["data"]["image_size"]
    augmentation = config["data"]["augmentation"]

    # TODO: Create training and validation datasets.
    #   Use PlantWildDataset with the appropriate root_dir
    #   (os.path.join with "train" or "val") and transforms.
    # Check the config file for specific keywords related to root directory and datasets
    train_dataset = None
    val_dataset = None

    # TODO: Create DataLoaders for both datasets.
    #   Use batch_size and num_workers from config["data"].
    #   Training: shuffle=True. Validation: shuffle=False.
    train_loader = None
    val_loader = None

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # TODO: Instantiate the model.
    #   Use PlantCNN with num_classes and dropout from config.
    #   Move to device.
    model = None

    print(f"Model parameters: {count_parameters(model):,}")

    # TODO: Define the loss function — nn.CrossEntropyLoss().
    criterion = None

    # TODO: Define the optimizer based on config["training"]["optimizer"].
    #   If "adam", use torch.optim.Adam.
    #   If "sgd", use torch.optim.SGD with momentum=0.9.
    #   Pass learning_rate and weight_decay from config.
    optimizer = None

    # TODO: Log sample images to TensorBoard for visual verification.
    #   Fetch one batch from train_loader using next(iter(...)).
    #   Log images using writer.add_images("sample_training_images", images, 0).


    # --- Training Loop ---
    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # TODO: Log four scalar metrics to TensorBoard using writer.add_scalar():
        #   "Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val"
        #   Use epoch as the global_step.


        # TODO: Save checkpoint if validation accuracy improved.
        #   Compare val_acc with best_val_acc. If better, update best_val_acc
        #   and call save_checkpoint(). Save to the path built from
        #   config["checkpoint"]["save_dir"] and experiment_name + "_best.pt".


    # TODO: Close the TensorBoard writer.

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantCNN")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config)
