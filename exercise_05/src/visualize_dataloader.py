"""
Exercise 05 - Verify Detection DataLoader
=============================================
Fully coded — no TODOs. Run this after completing dataset.py and
transforms.py to visually confirm the dataloader is working correctly.

Draws ground truth bounding boxes on sample images and saves a grid.
Check that:
  - Boxes align with the actual objects (not shifted or scaled wrong)
  - Class labels are correct (green = crop, red = weed)
  - Flipped images have flipped boxes (if augmentation is on)

Usage:
    python src/visualize_dataloader.py --config config.yaml
    python src/visualize_dataloader.py --config config.yaml --num_images 8
    python src/visualize_dataloader.py --config config.yaml --split val
"""

import os
import sys
import argparse

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PhenoBenchDetectionDataset
from transforms import get_train_transforms, get_val_transforms
from utils import collate_fn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_INFO = {
    1: {"name": "crop", "color": "lime"},
    2: {"name": "weed", "color": "red"},
}


def denormalize(image_tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    return img.clamp(0, 1)


def draw_boxes(ax, boxes, labels):
    """Draw bounding boxes with class-colored borders and labels."""
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        label_id = label.item()
        info = CLASS_INFO.get(label_id, {"name": f"cls_{label_id}", "color": "yellow"})

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=info["color"], facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, info["name"],
            color=info["color"], fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize detection dataloader")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=6,
                        help="Number of images to display")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Which split to visualize")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    data_root = config["data"]["root_dir"]

    # Use train transforms for train split (shows augmentation), val for val
    if args.split == "train":
        transforms = get_train_transforms()
    else:
        transforms = get_val_transforms()

    dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, args.split),
        annotation_file=os.path.join(data_root, "annotations", f"{args.split}.json"),
        transforms=transforms,
    )

    print(f"Split: {args.split}")
    print(f"Total images: {len(dataset)}")
    print(f"Categories: {dataset.categories}")

    num_images = min(args.num_images, len(dataset))
    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_images):
        image, target = dataset[i]

        # Denormalize for display
        img_display = denormalize(image).permute(1, 2, 0).numpy()

        ax = axes[i]
        ax.imshow(img_display)
        draw_boxes(ax, target["boxes"], target["labels"])

        n_objects = len(target["boxes"])
        n_crop = (target["labels"] == 1).sum().item()
        n_weed = (target["labels"] == 2).sum().item()
        ax.set_title(f"img_id={target['image_id'].item()} | "
                     f"{n_objects} objects ({n_crop} crop, {n_weed} weed)",
                     fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Detection DataLoader — {args.split} split "
        f"({'with augmentation' if args.split == 'train' else 'no augmentation'})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("outputs/visualizations", exist_ok=True)
    save_path = f"outputs/visualizations/dataloader_{args.split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {save_path}")
    print("Check that:")
    print("  - Boxes align with actual plants (not shifted)")
    print("  - Green = crop, Red = weed")
    print("  - Flipped images have correctly flipped boxes (train split)")


if __name__ == "__main__":
    main()
