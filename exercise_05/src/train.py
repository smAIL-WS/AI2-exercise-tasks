"""
Exercise 05 - Detection Training Script
==========================================
Training loop for object detection. Key differences from classification:
  - Model computes loss internally (returns loss_dict in train mode).
  - No explicit criterion needed.
  - Validation uses COCO mAP, not simple accuracy.

Usage:
    python src/train.py --config config.yaml
"""

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchDetectionDataset
from transforms import get_train_transforms, get_val_transforms
from model import get_detection_model, count_parameters
from evaluate import evaluate, print_results
from utils import set_seed, collate_fn, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, device, log_interval=10):
    """
    Train for one epoch. The model returns a loss dict in training mode.

    Returns:
        float: Average total loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):

        # TODO: Complete one training step.
        #
        #   Step 1: Move images to the device. Unlike classification where
        #           images are a single stacked tensor, here images is a
        #           tuple of individual tensors (different sizes possible).
        #           Use a list comprehension to move each image separately.
        images = None
        #
        #   Step 2: Move targets to the device. Each target is a dict of
        #           tensors. Use a list comprehension that loops over each
        #           target dict and moves every value to the device.
        targets = None
        #
        #   Step 3: Forward pass. In training mode, pass BOTH images and
        #           targets to the model. It returns a loss_dict. The keys
        #           differ by model (CNN models return 4 losses, RT-DETR
        #           returns 1), but the code handles all of them the same way.
        loss_dict = None
        #
        #   Step 4: Compute total_loss by summing all values in loss_dict (sum()).
        total_loss = None
        #
        #   Step 5: Zero gradients and call total_loss.backward()


        # Gradient clipping — prevents NaN from degenerate boxes or
        # transformer instability. Standard practice for detection training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # --- Logging (do not modify) ---
        running_loss += total_loss.item()
        total_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            detail = ", ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items())
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} ({detail})")

    avg_loss = running_loss / total_batches
    return avg_loss


def main(config_path):
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
    writer = SummaryWriter(log_dir=log_dir)

    data_root = config["data"]["root_dir"]

    # TODO: Create train and val datasets.
    #
    #   Step 1: Instantiate PhenoBenchDetectionDataset for training.
    #           root_dir should point to the "train" subfolder.
    #           annotation_file should point to "annotations/train.json".
    #           Apply get_train_transforms().
    #
    #   Step 2: Same for validation — "val" subfolder, "annotations/val.json",
    #           get_val_transforms().
    train_dataset = None
    val_dataset = None

    # TODO: Create DataLoaders for both datasets.
    #
    #   Step 1: Use batch_size and num_workers from config["data"].
    #
    #   Step 2: You MUST pass collate_fn=collate_fn (imported from utils).
    #           Without it, the default collate will crash because different
    #           images have different numbers of objects (different target
    #           tensor sizes). The custom collate_fn returns tuples instead
    #           of stacking.
    #
    #   Step 3: Train loader: shuffle=True. Val loader: shuffle=False.
    train_loader = None
    val_loader = None

    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # TODO: Create the detection model.
    #
    #   Step 1: Call get_detection_model() with name, num_classes, and
    #           pretrained from config["model"].
    #
    #   Step 2: Move to device.
    model = None

    print(f"Model: {config['model']['name']} | Params: {count_parameters(model):,}")

    # TODO: Create the optimizer.
    #
    #   Step 1: Collect only parameters that require gradients:
    #           [p for p in model.parameters() if p.requires_grad]
    #
    #   Step 2: Read the optimizer name from config["training"]. Default
    #           to "sgd" if not specified. Use .get("optimizer", "sgd").
    #
    #   Step 3: If "adamw", create torch.optim.AdamW with lr and
    #           weight_decay from config. AdamW is required for
    #           transformer models (RT-DETR).
    #           If "sgd", create torch.optim.SGD with lr, momentum,
    #           and weight_decay. SGD is the default for CNN models.
    optimizer = None

    # --- LR Scheduler (do not modify) ---
    num_epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- Training Loop ---
    best_map = 0.0

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{num_epochs}] (lr={current_lr:.6f})")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Step the scheduler after each epoch
        scheduler.step()

        # Evaluate
        metrics = evaluate(
        model, val_loader, device,
        annotation_file=os.path.join(data_root, "annotations", "val.json"),
    )
        current_map = metrics["ap50_95"]
        ap50 = metrics["ap50"]

        print(f"  Val mAP@[.50:.95]: {current_map:.4f}")
        print(f"  Val AP@.50:        {ap50:.4f}")

        # TODO: Log metrics to TensorBoard.
        #
        #   Use writer.add_scalar() to log four values:
        #   - "Loss/train" with train_loss
        #   - "mAP/val_AP50-95" with current_map
        #   - "mAP/val_AP50" with ap50
        #   - "LR" with current_lr
        #   Use epoch as the global_step argument.


        # TODO: Save checkpoint if mAP improved.
        #
        #   Step 1: Check if current_map is greater than best_map.
        #   Step 2: If yes, update best_map and call save_checkpoint()
        #           with the model, optimizer, epoch, and a metrics dict.
        #           Save to: config["checkpoint"]["save_dir"] / experiment_name + "_best.pt"
        #   Step 3: Print a message confirming the save.


    writer.close()
    print(f"\nTraining complete. Best mAP@[.50:.95]: {best_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
