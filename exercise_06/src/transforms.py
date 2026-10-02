"""
Exercise 06 - Segmentation Transforms
========================================
Transforms must apply spatial operations (flip, resize) to BOTH image and
mask. Color augmentations (jitter, normalize) apply to image ONLY.
Mask resizing MUST use nearest-neighbor interpolation.

Run this script to verify:
    python src/transforms.py
"""

import torch
import numpy as np
import torchvision.transforms.functional as F
import random


class Compose:
    """Chain multiple segmentation transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToTensor:
    """Convert PIL image to tensor and mask to long tensor."""

    def __call__(self, image, mask):
        # TODO: Convert image and mask to tensors.
        #
        #   Step 1: Convert the image using F.to_tensor() — scales pixels
        #           to [0, 1] and reorders to (C, H, W).
        #
        #   Step 2: Convert the mask to a long tensor. The mask is a PIL
        #           image with integer class indices (0, 1, 2). Convert
        #           via numpy first, then to a torch tensor with dtype
        #           int64. Do NOT use F.to_tensor() on the mask — that
        #           would scale the class indices to [0, 1] (insted, use 
        #           torch.as_tensor).

        return image, mask


class Normalize:
    """Normalize image only. Mask is unchanged."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class Resize:
    """Resize both image and mask to the given size."""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        # TODO: Resize both image and mask.
        #
        #   Step 1: Resize the image using F.resize() with default
        #           (bilinear) interpolation.
        #
        #   Step 2: Resize the mask using F.resize() with NEAREST
        #           interpolation. Bilinear would average neighboring
        #           class indices, creating invalid values like 0.5.
        #           Use F.InterpolationMode.NEAREST.
        #           If mask is a tensor, it needs a channel dimension
        #           for F.resize — unsqueeze(0) before, squeeze(0) after.

        return image, mask


class RandomHorizontalFlip:
    """Flip both image and mask horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        # TODO: With probability self.prob, flip both image and mask.
        #
        #   Unlike detection, no coordinate math needed — just apply
        #   F.hflip() to both. The mask values (class indices) stay the
        #   same; only their spatial positions change.

        return image, mask


class RandomVerticalFlip:
    """Flip both image and mask vertically."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        # TODO: Same pattern as RandomHorizontalFlip, using F.vflip().

        return image, mask


class ColorJitter:
    """Apply color jitter to the image ONLY. Mask is unchanged."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        from torchvision import transforms
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


def get_train_transforms(image_size):
    """Training transforms with augmentation."""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size):
    """Validation transforms — deterministic."""
    return Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    from PIL import Image

    size = 256
    dummy_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    dummy_mask = Image.fromarray(np.random.choice([0, 1, 2], size=(512, 512)).astype(np.uint8))

    print(f"Original image size: {dummy_img.size}")
    print(f"Original mask size: {dummy_mask.size}")
    print(f"Unique mask values (before): {np.unique(np.array(dummy_mask))}")

    train_tf = get_train_transforms(size)
    img_out, mask_out = train_tf(dummy_img.copy(), dummy_mask.copy())

    print(f"\nAfter train transforms:")
    print(f"  Image shape: {img_out.shape}")
    print(f"  Mask shape: {mask_out.shape}")
    print(f"  Mask dtype: {mask_out.dtype}")
    print(f"  Unique mask values: {torch.unique(mask_out).tolist()}")

    val_tf = get_val_transforms(size)
    img_out, mask_out = val_tf(dummy_img.copy(), dummy_mask.copy())

    print(f"\nAfter val transforms:")
    print(f"  Image shape: {img_out.shape}")
    print(f"  Mask shape: {mask_out.shape}")
    print(f"  Unique mask values: {torch.unique(mask_out).tolist()}")

    print("\n--- Transforms verified. ---")
