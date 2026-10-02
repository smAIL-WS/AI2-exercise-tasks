"""
Exercise 06 - Utility Functions (Fully Coded)
================================================
Helpers: seed setting, checkpointing, mask visualization.
"""

import os
import random

import numpy as np
from PIL import Image
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (46, 204, 113),   # crop — green
    2: (231, 76, 60),    # weed — red
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model and optimizer state."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load a checkpoint and restore model state."""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def colorize_mask(mask):
    """Convert a (H, W) class mask to (H, W, 3) RGB for visualization."""
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask_np == class_id] = color
    return rgb


def overlay_mask(image, mask, alpha=0.55):
    """Blend colorized mask onto a normalized image for inspection."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_01 = (image.detach().cpu() * std + mean).clamp(0, 1)
    img_np = (img_01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    rgb_mask = colorize_mask(mask)
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    fg = mask_np != 0

    blended = img_np.copy()
    blended[fg] = (img_np[fg] * (1 - alpha) + rgb_mask[fg] * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)
