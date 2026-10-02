"""
Exercise 07 - Few-Shot Finetuning Training Script
=====================================================
Same loop shape as Exercise 06, tuned for a tiny dataset: the model
computes its own loss when given labels (like Ex05's detector, unlike
Ex06's explicit criterion), and the optimizer only ever sees parameters
with requires_grad=True.

Usage:
    python src/train.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchFewShotSegDataset
from transforms import get_train_transforms, get_val_transforms
from model import load_pretrained_segformer, freeze_encoder, parameter_summary, print_parameter_summary
from evaluate import evaluate
from utils import set_seed, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, device, log_interval=5):
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
        #   Step 2: Zero gradients.
        #
        #   Step 3: Forward pass WITH labels: model(pixel_values=images,
        #           labels=masks). Unlike Ex06, there is no separate
        #           criterion -- SegFormer computes cross-entropy loss
        #           internally (upsampling its logits to the mask's
        #           resolution first) and returns it as outputs.loss.
        #
        #   Step 4: Backward and optimizer step using outputs.loss.


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
    #
    #   Training: PhenoBenchFewShotSegDataset with root_dir pointing at
    #   the "train" subfolder, num_shots=config["data"]["num_shots"],
    #   seed=config["seed"], and train transforms.
    #
    #   Validation: same class, root_dir="val", num_shots=None (use the
    #   whole split), and val transforms.
    train_dataset = None
    val_dataset = None

    # TODO: Create DataLoaders.
    #   No custom collate_fn needed -- default stacking works.
    #   Use batch_size and num_workers from config.
    train_loader = None
    val_loader = None

    print(f"Few-shot train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # TODO: Build the model.
    #
    #   Step 1: load_pretrained_segformer() with model.name and
    #           model.num_classes from config.
    #
    #   Step 2: If config["model"]["freeze_encoder"] is True, apply
    #           freeze_encoder().
    #
    #   Step 3: Move the model to device.
    model = None

    summary = parameter_summary(model)
    print_parameter_summary(summary)

    # TODO: Build the optimizer.
    #
    #   Only parameters with requires_grad=True should be passed to the
    #   optimizer -- passing frozen parameters wastes memory on unused
    #   optimizer state (e.g. AdamW's per-parameter moment buffers) and
    #   provides no benefit, since their gradients are never computed.
    #   Use AdamW with lr and weight_decay from config, and
    #   filter(lambda p: p.requires_grad, model.parameters()) as the
    #   parameter list.
    optimizer = None

    # --- Training Loop ---
    best_miou = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
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
    parser = argparse.ArgumentParser(description="Few-shot finetune a segmentation model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
