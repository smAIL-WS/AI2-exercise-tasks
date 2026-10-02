"""
Optional / Take-Home — Partial Unfreezing
=============================================
Not required. For students who finish early, or as a take-home extension.

Section 4 (required) freezes the entire SegFormer encoder and trains only
the decode head. This script goes one step further: it also unfreezes the
LAST encoder stage, trading a higher overfitting risk for more capacity to
adapt to PhenoBench.

SegFormer's internal stage naming has changed across `transformers`
versions (this changed as recently as the 5.x line), so this script
detects which naming scheme is present at runtime instead of assuming one
-- see find_last_stage_prefix() below.

Requires Exercise A-C (`src/model.py`, `src/dataset.py`, `src/train.py`)
already completed, since it reuses those functions directly.

Usage:
    python optional/partial_unfreeze.py --config ../config.yaml
"""

import os
import sys
import argparse

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset import PhenoBenchFewShotSegDataset
from transforms import get_train_transforms, get_val_transforms
from model import load_pretrained_segformer, parameter_summary, print_parameter_summary
from train import train_one_epoch
from evaluate import evaluate
from utils import set_seed, save_checkpoint


def find_last_stage_prefix(model, num_stages=4):
    """
    Detect the naming scheme SegFormer's encoder uses in the installed
    `transformers` version, and return how to select the last stage's
    parameters.

    Tries known naming schemes in order and falls back to a positional
    heuristic (the last ~1/num_stages of the encoder's named parameters,
    by position) if none match -- so this keeps working even if the
    internal naming changes again in a future version.

    Returns:
        ("substring", prefix) or ("positional", num_tensors_to_unfreeze)
    """
    names = [name for name, _ in model.segformer.named_parameters()]
    last_idx = num_stages - 1

    candidates = [
        f"encoder.block.{last_idx}",   # older `transformers` versions
        f"stages.{last_idx}",          # newer `transformers` versions
    ]
    for prefix in candidates:
        if any(prefix in name for name in names):
            return ("substring", prefix)

    num_to_unfreeze = max(1, len(names) // num_stages)
    return ("positional", num_to_unfreeze)


def unfreeze_last_stage(model, num_stages=4):
    """Freeze everything except the head, then also unfreeze the last
    encoder stage (detected via find_last_stage_prefix)."""
    for param in model.segformer.parameters():
        param.requires_grad = False
    for param in model.decode_head.parameters():
        param.requires_grad = True

    mode, value = find_last_stage_prefix(model, num_stages)
    if mode == "substring":
        for name, param in model.segformer.named_parameters():
            if value in name:
                param.requires_grad = True
        print(f"Unfroze last encoder stage by name (matched '{value}').")
    else:
        params = list(model.segformer.named_parameters())
        for name, param in params[-value:]:
            param.requires_grad = True
        print(f"Unfroze the last {value} encoder parameter tensors by "
              f"position (no known naming scheme matched -- see "
              f"find_last_stage_prefix()).")

    return model


def main(config_path):
    config = yaml.safe_load(open(config_path))
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    data_root = config["data"]["root_dir"]
    image_size = config["data"]["image_size"]

    train_dataset = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(data_root, "train"),
        transforms=get_train_transforms(image_size),
        num_shots=config["data"]["num_shots"],
        seed=config["seed"],
    )
    val_dataset = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(data_root, "val"),
        transforms=get_val_transforms(image_size),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    model = load_pretrained_segformer(config["model"]["name"], config["model"]["num_classes"])
    model = unfreeze_last_stage(model)
    model = model.to(device)

    print_parameter_summary(parameter_summary(model))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_miou = 0.0
    num_epochs = config["training"]["epochs"]
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )
        metrics = evaluate(model, val_loader, device, config["model"]["num_classes"])
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}  Val mIoU: {metrics['miou']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            ckpt_path = os.path.join(
                config["checkpoint"]["save_dir"],
                config["experiment_name"] + "_partial_unfreeze_best.pt",
            )
            save_checkpoint(model, optimizer, epoch, {"miou": best_miou}, ckpt_path)

    print(f"\nBest mIoU with the last encoder stage unfrozen: {best_miou:.4f}")
    print("Compare this against your head-only run from Exercise E.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    args = parser.parse_args()
    main(args.config)
