"""
Exercise 07 - Segmentation Transforms (Fully Coded)
=======================================================
Identical to Exercise 06's transforms.py, reused unchanged: SegFormer
expects the same ImageNet-normalized RGB input as every other model in
this course. Transforms must apply spatial operations (flip, resize) to
BOTH image and mask. Color augmentations (jitter, normalize) apply to the
image only, and mask resizing MUST use nearest-neighbor interpolation.
"""

import random

import torch
import numpy as np
import torchvision.transforms.functional as F


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
        image = F.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
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
        image = F.resize(image, self.size)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
            mask = mask.squeeze(0)
        else:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    """Flip both image and mask horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    """Flip both image and mask vertically."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.vflip(image)
            mask = F.vflip(mask)
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
    """Validation/test transforms — deterministic."""
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

    size = 512
    dummy_img = Image.new("RGB", (1024, 1024), color=(100, 150, 200))
    dummy_mask = Image.fromarray(np.random.choice([0, 1, 2], size=(1024, 1024)).astype(np.uint8))

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
