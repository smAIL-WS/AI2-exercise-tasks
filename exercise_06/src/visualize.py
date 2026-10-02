"""
Exercise 06 - Dataset Visualization
=======================================
Standalone script that saves a grid of [image | colorized mask | overlay]
for a handful of samples, so you can actually look at what you're training
on. Given fully implemented -- no TODOs here, just run it once dataset.py's
AND transforms.py's TODOs are done:

    python src/visualize.py --split train --n 6

Add --augment to also show what RandomHorizontalFlip does -- a 4th column
with the SAME sample, guaranteed-flipped (not randomly, so you always see
the effect, not a coin flip), to confirm image and mask move together:

    python src/visualize.py --split train --n 6 --augment

Output: outputs/visualizations/{split}_samples.png
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchSegmentationDataset
from transforms import get_val_transforms
from utils import colorize_mask, overlay_mask, IMAGENET_MEAN, IMAGENET_STD


def get_flip_demo_transform(image_size):
    """
    Same steps as transforms.py's get_train_transforms(), except the flip
    always happens (p=1.0) instead of 50% of the time -- so --augment always
    shows you the effect instead of leaving it to chance.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.Resize((image_size, image_size)),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64,
                    "others": None}, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def unnormalize(image):
    """Undo transforms.py's Normalize so the image is displayable again."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_grid(dataset, raw_dataset, indices, image_size, augment=False, cell=256, pad=8, header_h=28):
    """
    Build a [image | colorized mask | overlay] grid, one row per index.
    With augment=True, adds a 4th column: the same sample with a guaranteed
    horizontal flip, read fresh from raw_dataset (untransformed) so it's
    independent of whatever `dataset`'s own transform did.
    """
    columns = ["image", "mask (colorized)", "overlay"]
    if augment:
        columns.append("overlay, flipped")
    n_cols = len(columns)
    n = len(indices)
    width = cell * n_cols + pad * (n_cols + 1)
    height = header_h + cell * n + pad * (n + 1)
    grid = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for col, name in enumerate(columns):
        x = pad + col * (cell + pad)
        draw.text((x, 6), name, fill=(0, 0, 0), font=font)

    flip_transform = get_flip_demo_transform(image_size) if augment else None

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        img_panel = Image.fromarray(unnormalize(image))
        mask_panel = Image.fromarray(colorize_mask(mask))
        overlay_panel = overlay_mask(image, mask)

        y = header_h + pad + row * (cell + pad)
        grid.paste(img_panel, (pad, y))
        grid.paste(mask_panel, (pad * 2 + cell, y))
        grid.paste(overlay_panel, (pad * 3 + cell * 2, y))

        if augment:
            raw_image, raw_mask = raw_dataset[idx]
            flipped_image, flipped_mask = flip_transform(raw_image, raw_mask)
            flipped_panel = overlay_mask(flipped_image, flipped_mask)
            grid.paste(flipped_panel, (pad * 4 + cell * 3, y))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize PhenoBench segmentation samples")
    parser.add_argument("--data-root", type=str, default="data/phenobench")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=6, help="Number of samples to show")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true",
                         help="Add a 4th column showing each sample with a guaranteed "
                              "horizontal flip, to check image+mask move together")
    args = parser.parse_args()

    dataset = PhenoBenchSegmentationDataset(
        root_dir=args.data_root, split=args.split,
        transform=get_val_transforms(args.image_size),  # deterministic, no random augmentation
    )
    # Raw (untransformed) view of the same data -- only used for --augment's
    # guaranteed-flip column, kept separate so it can't affect `dataset`.
    raw_dataset = PhenoBenchSegmentationDataset(
        root_dir=args.data_root, split=args.split, transform=None,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False)

    grid = build_grid(dataset, raw_dataset, indices, args.image_size, augment=args.augment)

    out_dir = "outputs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_samples.png")
    grid.save(out_path)
    print(f"Saved {len(indices)} {args.split} samples to {out_path}")
    if args.augment:
        print("4th column ('overlay, flipped') is the same sample with a guaranteed "
              "horizontal flip -- confirm the mask lines up with the flipped image.")


if __name__ == "__main__":
    main()
