"""
Exercise 05 - Detection Transforms
=====================================
Transforms for object detection must handle both the image AND the
bounding boxes. If an image is flipped, the boxes must be flipped too.

Run this script to verify:
    python src/transforms.py
"""

import torch
import torchvision.transforms.functional as F
import random


class Compose:
    """Chain multiple detection transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to tensor. Boxes remain unchanged."""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image tensor. Boxes remain unchanged."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """
    Randomly flip image and boxes horizontally with given probability.
    Boxes are in [x1, y1, x2, y2] format.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        # TODO: Flip the image and mirror box x-coordinates.
        #
        #   Step 1: Generate a random number (random.random()) and only
        #           proceed if it falls below self.prob.
        #
        #   Step 2: Flip the image using F.hflip().
        #
        #   Step 3: Get the image width. Note that the image may be a
        #           PIL image or a tensor at this point depending on
        #           transform ordering — check which type it is and
        #           use .width (PIL) or .shape[-1] (tensor) accordingly.
        #
        #   Step 4: Mirror the bounding boxes. Clone the boxes first
        #           to avoid modifying the original. Only x-coordinates
        #           change — y stays the same:
        #               new_x1 = width - old_x2
        #               new_x2 = width - old_x1
        #           Hint: boxes[:, 0] is all x1 values, boxes[:, 2] is
        #           all x2 values.
        #
        #   Step 5: Update target["boxes"] with the flipped boxes.

        return image, target


class Resize:
    """
    Resize image so the shorter side equals min_size.
    Scale all bounding box coordinates proportionally.
    """

    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size or min_size

    def __call__(self, image, target):
        # TODO: Resize the image and scale boxes to match.
        #
        #   Step 1: Get the original height and width. Same type check
        #           as in RandomHorizontalFlip — PIL uses .size (returns
        #           w, h), tensor uses .shape[-2:] (returns h, w).
        #
        #   Step 2: Compute the scale factor so the shorter side becomes
        #           self.min_size. If self.max_size is set, ensure the
        #           longer side doesn't exceed it.
        #
        #   Step 3: Compute new_h and new_w by multiplying originals
        #           by the scale factor. Resize the image using
        #           F.resize(image, [new_h, new_w]).
        #
        #   Step 4: Compute separate scale factors for x and y:
        #               scale_x = new_w / old_w
        #               scale_y = new_h / old_h
        #           Multiply all box x-coordinates (columns 0 and 2)
        #           by scale_x, and y-coordinates (columns 1 and 3)
        #           by scale_y.
        #
        #   Step 5: Recompute target["area"] from the scaled boxes:
        #           area = (x2 - x1) * (y2 - y1).

        return image, target


def get_train_transforms():
    """Training transforms with augmentation."""
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(prob=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation transforms — deterministic, no augmentation."""
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    from PIL import Image

    dummy_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    dummy_target = {
        "boxes": torch.tensor([[100, 200, 300, 400], [500, 600, 700, 800]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "area": torch.tensor([200 * 200, 200 * 200], dtype=torch.float32),
        "iscrowd": torch.tensor([0, 0], dtype=torch.uint8),
        "image_id": torch.tensor([1]),
    }

    print("Original boxes:", dummy_target["boxes"])

    train_tf = get_train_transforms()
    img_out, tgt_out = train_tf(dummy_img, dummy_target.copy())
    print(f"After train transforms — image shape: {img_out.shape}, boxes: {tgt_out['boxes']}")

    val_tf = get_val_transforms()
    img_out, tgt_out = val_tf(dummy_img, dummy_target.copy())
    print(f"After val transforms — image shape: {img_out.shape}, boxes: {tgt_out['boxes']}")

    print("\n--- Transforms verified. ---")
