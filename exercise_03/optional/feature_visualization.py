"""
Optional — Feature Map and Filter Visualization
==================================================
Demonstrates how to visualize:
  1. Learned convolutional filters (the actual weights)
  2. Feature maps (activations) produced when an image passes through
     each convolutional block

Uses PyTorch forward hooks to capture intermediate activations without
modifying the model's forward() method.

This is a complete script for instructor demonstration.

Usage:
    python optional/feature_visualization.py --config config.yaml \
        --checkpoint outputs/checkpoints/plantwild_baseline_best.pt \
        --image_path data/plantwild/test/apple_scab/001.jpg
"""

import os
import sys
import argparse

import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import PlantCNN
from transforms import get_val_transforms
from utils import load_checkpoint


def visualize_filters(model, save_path="outputs/filters.png"):
    """
    Visualize the first convolutional layer's learned filters.

    The first conv layer has filters of shape (32, 3, 3, 3) — 32 filters,
    each with 3 channels (RGB), 3x3 spatial. We visualize each filter
    as a small RGB image.
    """
    # Get first conv layer's weights
    first_conv = None
    for module in model.features.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        print("No Conv2d layer found.")
        return

    filters = first_conv.weight.data.cpu().clone()
    num_filters = filters.shape[0]

    # Normalize each filter to [0, 1] for display
    filters -= filters.min()
    filters /= filters.max() + 1e-8

    # Plot in a grid
    cols = 8
    rows = (num_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle("First Conv Layer — Learned Filters", fontsize=14)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        if i < num_filters:
            # filters[i] is (3, 3, 3) — permute to (3, 3, 3) for imshow (HWC)
            img = filters[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Filters saved to: {save_path}")


def visualize_feature_maps(model, image_tensor, device, save_path="outputs/feature_maps.png"):
    """
    Visualize feature maps (activations) at each conv block using forward hooks.

    A forward hook is a function that gets called every time a module's
    forward() runs. We register hooks on each MaxPool2d layer to capture
    the output after each conv block.
    """
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().cpu()
        return hook_fn

    # Register hooks on MaxPool2d layers (end of each block)
    hooks = []
    pool_idx = 0
    for module in model.features.modules():
        if isinstance(module, torch.nn.MaxPool2d):
            pool_idx += 1
            name = f"Block {pool_idx}"
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        _ = model(image_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Plot first 16 feature maps from each block
    num_blocks = len(activations)
    num_maps = 16
    fig, axes = plt.subplots(num_blocks, num_maps, figsize=(num_maps * 1.2, num_blocks * 1.5))
    fig.suptitle("Feature Maps per Conv Block", fontsize=14)

    for row, (name, fmaps) in enumerate(sorted(activations.items())):
        fmaps = fmaps[0]  # remove batch dim → (C, H, W)
        for col in range(num_maps):
            ax = axes[row, col] if num_blocks > 1 else axes[col]
            if col < fmaps.shape[0]:
                ax.imshow(fmaps[col].numpy(), cmap="viridis")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=50)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature maps saved to: {save_path}")


def main(config_path, checkpoint_path, image_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Load model
    model = PlantCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, model)
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(val_acc={ckpt['val_accuracy']:.4f})")
    else:
        print("No checkpoint provided — using random weights.")

    # Visualize filters
    visualize_filters(model, save_path="outputs/filters.png")

    # Load and transform one image
    image_size = config["data"]["image_size"]
    transform = get_val_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    # Visualize feature maps
    visualize_feature_maps(model, image_tensor, device,
                           save_path="outputs/feature_maps.png")

    print("\nDone. Check outputs/ for the saved figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image_path)
