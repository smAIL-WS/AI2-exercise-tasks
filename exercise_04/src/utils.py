"""
Exercise 03 - Utility Functions (SOLUTION)
============================================
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def save_checkpoint(model, optimizer, epoch, val_accuracy, filepath, model_name=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


if __name__ == "__main__":
    # Test set_seed
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b), "set_seed is not working correctly"
    print("set_seed: OK")

    # Test compute_accuracy
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
    targets = torch.tensor([0, 1, 2])
    acc = compute_accuracy(outputs, targets)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"compute_accuracy: OK (acc={acc})")

    # Test save/load checkpoint
    import torch.nn as nn
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, epoch=5, val_accuracy=0.85,
                    filepath="outputs/checkpoints/test_ckpt.pt")
    ckpt = load_checkpoint("outputs/checkpoints/test_ckpt.pt", model, optimizer)
    assert ckpt["epoch"] == 5
    assert ckpt["val_accuracy"] == 0.85
    print(f"save/load checkpoint: OK (epoch={ckpt['epoch']}, acc={ckpt['val_accuracy']})")

    os.remove("outputs/checkpoints/test_ckpt.pt")

    print("\n--- All utilities verified. ---")
