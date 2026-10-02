"""
Optional — Visualize Segmentation Predictions
=================================================
Load a trained checkpoint, run inference on validation images, and save
a side-by-side comparison: image | ground truth mask | predicted mask | overlay.

Green = crop, Red = weed, Black = background.

Usage:
    python optional/visualize_predictions.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_unet_best.pt

    python optional/visualize_predictions.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_unet_best.pt \
        --num_images 8 --split val
"""

import os
import sys
import argparse

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset import PhenoBenchSegDataset
from transforms import get_val_transforms
from model import get_segmentation_model
from utils import load_checkpoint, colorize_mask, overlay_mask


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["background", "crop", "weed"]
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (46, 204, 113),
    2: (231, 76, 60),
}


def denormalize(image_tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image_tensor.cpu() * std + mean).clamp(0, 1)


def compute_sample_iou(pred, target, num_classes):
    """Compute per-class IoU for a single image."""
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    valid = [v for v in ious if v == v]  # exclude nan
    mean_iou = sum(valid) / len(valid) if valid else 0.0
    return mean_iou, ious


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run training first: python src/train.py --config config.yaml")
        sys.exit(1)

    config = yaml.safe_load(open(args.config))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_root = config["data"]["root_dir"]
    image_size = config["data"]["image_size"]
    num_classes = config["model"]["num_classes"]

    # Load model
    model = get_segmentation_model(
        name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    model.eval()

    # Load dataset (no augmentation)
    dataset = PhenoBenchSegDataset(
        root_dir=os.path.join(data_root, args.split),
        transforms=get_val_transforms(image_size),
    )

    num_images = min(args.num_images, len(dataset))

    # 4 columns: image | GT mask | predicted mask | prediction overlay
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    print(f"Running inference on {num_images} {args.split} images...")

    for i in range(num_images):
        image, gt_mask = dataset[i]

        # Inference
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            if isinstance(output, dict):
                output = output["out"]
            pred_mask = output.argmax(dim=1).squeeze(0).cpu()

        # Compute per-sample mIoU
        miou, per_class = compute_sample_iou(pred_mask, gt_mask, num_classes)

        # Prepare visuals
        img_display = denormalize(image).permute(1, 2, 0).numpy()
        gt_rgb = colorize_mask(gt_mask)
        pred_rgb = colorize_mask(pred_mask)
        pred_overlay = np.array(overlay_mask(image, pred_mask))

        # Plot
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title("image", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("ground truth", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_rgb)
        iou_str = ", ".join(
            f"{CLASS_NAMES[c]}={v:.2f}" if v == v else f"{CLASS_NAMES[c]}=N/A"
            for c, v in enumerate(per_class)
        )
        axes[i, 2].set_title(f"prediction (mIoU={miou:.2f})", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_overlay)
        axes[i, 3].set_title("prediction overlay", fontsize=10)
        axes[i, 3].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=len(CLASS_NAMES), fontsize=11, frameon=False,
    )

    plt.suptitle(
        f"Segmentation Predictions — {args.split} split "
        f"({config['model']['name']})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    os.makedirs("outputs/visualizations", exist_ok=True)
    save_path = f"outputs/visualizations/seg_predictions_{args.split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
