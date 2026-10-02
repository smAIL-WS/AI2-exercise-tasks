"""
Optional — Visualize Detection Predictions
=============================================
Draw predicted and ground truth bounding boxes on images.
Green = ground truth, Red = predictions (with score).

Usage:
    python optional/visualize_predictions.py --config config.yaml \
        --checkpoint outputs/checkpoints/phenobench_fasterrcnn_best.pt \
        --num_images 5
"""

import os
import sys
import argparse

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset import PhenoBenchDetectionDataset
from transforms import get_val_transforms, ToTensor, Normalize
from model import get_detection_model
from utils import load_checkpoint


CLASS_NAMES = {1: "crop", 2: "weed"}
COLORS = {1: "lime", 2: "red"}


def visualize(image_pil, gt_target, predictions, score_threshold=0.5,
              save_path=None):
    """Draw GT (green) and predicted (red) boxes on an image."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_pil)

    # Ground truth boxes (green dashed)
    if gt_target is not None:
        for box, label in zip(gt_target["boxes"], gt_target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            cls_name = CLASS_NAMES.get(label.item(), "?")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT: {cls_name}", color="lime", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Predicted boxes (colored by class)
    for box, label, score in zip(predictions["boxes"], predictions["labels"],
                                  predictions["scores"]):
        if score.item() < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        cls_name = CLASS_NAMES.get(label.item(), "?")
        color = COLORS.get(label.item(), "yellow")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y2 + 15, f"{cls_name}: {score.item():.2f}", color=color,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    ax.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main(config_path, checkpoint_path, num_images=5, score_threshold=0.5):
    config = yaml.safe_load(open(config_path))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_root = config["data"]["root_dir"]

    # Load model
    model = get_detection_model(
        num_classes=config["model"]["num_classes"], pretrained=False,
    ).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Dataset without normalization (for display), but with ToTensor for model
    dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, "val"),
        annotation_file=os.path.join(data_root, "annotations", "val.json"),
        transforms=None,
    )

    inference_transforms = torch.nn.Sequential()  # just ToTensor for model input

    print(f"Visualizing {num_images} images (score threshold: {score_threshold})")

    for i in range(min(num_images, len(dataset))):
        image_pil, target = dataset[i]

        # Convert to tensor for model
        import torchvision.transforms.functional as TF
        image_tensor = TF.to_tensor(image_pil).to(device)

        with torch.no_grad():
            predictions = model([image_tensor])[0]

        # Move predictions to CPU
        predictions = {k: v.cpu() for k, v in predictions.items()}

        save_path = f"outputs/visualizations/prediction_{i:03d}.png"
        visualize(image_pil, target, predictions,
                  score_threshold=score_threshold, save_path=save_path)
        print(f"  Saved: {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.num_images, args.score_threshold)
