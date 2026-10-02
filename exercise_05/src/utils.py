"""
Exercise 05 - Utility Functions (Fully Coded)
================================================
Detection-specific helpers: custom collate_fn, seed setting, checkpointing.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    Custom collate function for detection.

    Default collate tries to stack all tensors, which fails because
    different images have different numbers of objects (different-sized
    target tensors). This function returns tuples of lists instead.

    Args:
        batch: list of (image, target) tuples from the Dataset.

    Returns:
        tuple: (tuple of images, tuple of target dicts)
    """
    return tuple(zip(*batch))


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
