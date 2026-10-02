#!/usr/bin/env python3
"""
One-off script that generates the illustrative figures embedded in
exercise_06/README.md (a real mask example, pixel-wise IoU, mIoU good vs.
bad prediction, and a U-Net architecture schematic). Not part of the graded
exercise -- just documentation tooling. Needs data/phenobench symlinked (see
README) since two of the four figures use a real PhenoBench image/mask pair
(the IoU and U-Net schematic figures are synthetic/abstract and don't).
Regenerate with:

    python assets/generate_figures.py
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path(__file__).parent
DATA_ROOT = OUT_DIR.parent / "data" / "phenobench"
SAMPLE_FILE = "05-15_00241_P0030953.png"  # has substantial crop AND weed pixels

GREEN = "#2ECC71"   # crop
RED = "#E74C3C"      # weed
BLUE = "#4C72B0"
ORANGE = "#DD8452"
PURPLE = "#9467BD"
GRAY = "#555555"

CLASS_COLORS = {0: (0, 0, 0), 1: (46, 204, 113), 2: (231, 76, 60)}

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_sample():
    img = Image.open(DATA_ROOT / "images" / "val" / SAMPLE_FILE).convert("RGB")
    mask = np.array(Image.open(DATA_ROOT / "masks" / "val" / SAMPLE_FILE))
    return np.array(img), mask


def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def overlay(img, mask, alpha=0.55):
    rgb_mask = colorize(mask)
    fg = mask != 0
    blended = img.copy()
    blended[fg] = (img[fg] * (1 - alpha) + rgb_mask[fg] * alpha).astype(np.uint8)
    return blended


def dilate(mask_bool, iterations=1):
    m = mask_bool.copy()
    for _ in range(iterations):
        m = (
            m
            | np.roll(m, 1, axis=0) | np.roll(m, -1, axis=0)
            | np.roll(m, 1, axis=1) | np.roll(m, -1, axis=1)
        )
    return m


def erode(mask_bool, iterations=1):
    m = mask_bool.copy()
    for _ in range(iterations):
        m = (
            m
            & np.roll(m, 1, axis=0) & np.roll(m, -1, axis=0)
            & np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
        )
    return m


def pixel_iou(pred, target, num_classes=3):
    ious, valid = [], []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            iou = inter / union
            ious.append(iou)
            valid.append(iou)
    return (sum(valid) / len(valid) if valid else 0.0), ious


# ===========================================================================
# Figure 1: what a segmentation mask is (real data)
# ===========================================================================
def figure_mask_example():
    img, mask = load_sample()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    panels = [
        ("image", img),
        ("mask (colorized)", colorize(mask)),
        ("overlay", overlay(img, mask)),
    ]
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("An image, its mask, and the two blended together", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mask_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Figure 2: pixel-wise IoU (abstract, exact rasterized pixel counts)
# ===========================================================================
def figure_pixel_iou():
    grid = 300
    yy, xx = np.mgrid[0:grid, 0:grid]

    def circle(cx, cy, r):
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    r = 55
    cx_a, cy_a = 110, 150
    # Offsets found by numeric search so the displayed IoU actually lands
    # near the target (0.79, 0.50, 0.10) -- see the comment in the module
    # docstring: values are computed by rasterizing and counting pixels,
    # the same way compute_iou() works, not from a closed-form formula.
    cases = [
        ("High overlap", circle(cx_a, cy_a, r), circle(cx_a + 10, cy_a, r)),
        ("Right at the threshold", circle(cx_a, cy_a, r), circle(cx_a + 29, cy_a, r)),
        ("Low overlap", circle(cx_a, cy_a, r), circle(cx_a + 78, cy_a, r)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))
    for ax, (title, region_a, region_b) in zip(axes, cases):
        inter = region_a & region_b
        union = region_a | region_b
        iou = inter.sum() / union.sum()
        verdict = "match  ✓" if iou >= 0.5 else "no match  ✗"

        canvas = np.ones((grid, grid, 3))
        canvas[region_a] = np.array([76, 114, 176]) / 255    # box A blue, translucent look
        canvas[region_b] = np.array([221, 132, 82]) / 255    # box B orange
        canvas[inter] = np.array([148, 103, 189]) / 255      # intersection purple

        ax.imshow(canvas, origin="upper")
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor("#CCCCCC")

        color = "#2ca02c" if iou >= 0.5 else "#d62728"
        ax.text(grid / 2, grid + 16, f"IoU = {iou:.2f}  —  {verdict}",
                fontsize=12.5, color=color, ha="center", fontweight="bold")

    axes[0].legend(
        handles=[
            patches.Patch(facecolor=np.array([76, 114, 176]) / 255, label="region A (e.g. ground truth)"),
            patches.Patch(facecolor=np.array([221, 132, 82]) / 255, label="region B (e.g. prediction)"),
            patches.Patch(facecolor=np.array([148, 103, 189]) / 255, label="intersection"),
        ],
        loc="lower left", fontsize=8.5, framealpha=0.9,
    )
    fig.suptitle("Pixel-wise IoU: same formula as Ex05, now over pixel SETS instead of box areas",
                 fontsize=14.5, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pixel_iou_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Figure 3: mIoU -- good vs. bad prediction (real image, synthetic predictions)
# ===========================================================================
def figure_miou_good_bad():
    img, gt = load_sample()

    # "Good" prediction: minor boundary noise (1px erosion) on crop, weed
    # otherwise correct -- what a reasonably well-trained model looks like.
    crop_good = erode(gt == 1, iterations=1)
    weed_good = gt == 2
    good = np.zeros_like(gt)
    good[crop_good] = 1
    good[weed_good] = 2

    # "Bad" prediction: crop boundary crudely over-dilated (blob instead of
    # leaf shape) AND weed missed entirely -- the class-imbalance failure
    # mode the README warns about (weed is only ~0.2% of pixels; a model
    # trained without class weighting can get away with never predicting it).
    crop_bad = dilate(gt == 1, iterations=6)
    bad = np.zeros_like(gt)
    bad[crop_bad] = 1  # weed left at 0 (background) everywhere

    miou_good, _ = pixel_iou(good, gt)
    miou_bad, _ = pixel_iou(bad, gt)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8))
    panels = [
        ("ground truth", colorize(gt), None),
        (f"good prediction  (mIoU = {miou_good:.2f})", colorize(good), "#2ca02c"),
        (f"bad prediction  (mIoU = {miou_bad:.2f})", colorize(bad), "#d62728"),
    ]
    for ax, (title, arr, color) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(title, fontsize=12.5, color=color or "black")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("mIoU tells the two apart even though both \"found the crop row\"",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "miou_good_bad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Figure 4: PhenoSegNet (U-Net) architecture schematic
# ===========================================================================
def figure_unet_schematic():
    ENC, DEC, BOTTLE, SKIP = "#4C72B0", "#DD8452", "#9467BD", "#888888"
    box_w, box_h = 2.3, 0.95
    x_left, x_right, x_mid = 1.6, 8.0, 5.5
    head_x, head_w = 10.7, 1.8

    fig, ax = plt.subplots(figsize=(13.2, 7.5))
    ax.set_xlim(-1.8, 14.5)
    ax.set_ylim(0.2, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, label, color, w=box_w, h=box_h):
        for face, alpha, lw in [(color, 0.16, 0), ("none", 1, 1.6)]:
            ax.add_patch(patches.FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                linewidth=lw, edgecolor=color, facecolor=face, alpha=alpha, zorder=3,
            ))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color="#222", zorder=4)

    def arrow(xy_from, xy_to, color=GRAY, lw=2.0, ls="-", label=None, label_xy=None, label_color=None):
        ax.annotate("", xy=xy_to, xytext=xy_from, zorder=2,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                      shrinkA=2, shrinkB=2))
        if label:
            lx, ly = label_xy if label_xy else ((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2)
            ax.text(lx, ly, label, fontsize=8, color=label_color or color, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=4)

    y_levels = {256: 8.0, 128: 5.9, 64: 3.8, 32: 1.6}  # spatial resolution -> vertical position

    # --- Encoder column ---
    block(x_left, y_levels[256], "enc1\nConvBlock\n32ch · 256×256", ENC)
    block(x_left, y_levels[128], "enc2\nConvBlock\n64ch · 128×128", ENC)
    block(x_left, y_levels[64], "enc3\nConvBlock\n128ch · 64×64", ENC)
    arrow((x_left, y_levels[256] - box_h / 2), (x_left, y_levels[128] + box_h / 2), color=ENC,
          label="pool /2", label_xy=(x_left - 1.25, (y_levels[256] + y_levels[128]) / 2))
    arrow((x_left, y_levels[128] - box_h / 2), (x_left, y_levels[64] + box_h / 2), color=ENC,
          label="pool /2", label_xy=(x_left - 1.25, (y_levels[128] + y_levels[64]) / 2))

    # --- Bottleneck ---
    block(x_mid, y_levels[32], "bottleneck\nConvBlock\n256ch · 32×32", BOTTLE, w=2.8)
    arrow((x_left + 0.3, y_levels[64] - box_h / 2), (x_mid - 1.6, y_levels[32] + 0.42), color=ENC,
          label="pool /2", label_xy=(x_left + 1.0, (y_levels[64] + y_levels[32]) / 2 - 0.15))

    # --- Decoder column ---
    block(x_right, y_levels[64], "dec1\nConvBlock\n128ch · 64×64", DEC)
    block(x_right, y_levels[128], "dec2\nConvBlock\n64ch · 128×128", DEC)
    block(x_right, y_levels[256], "dec3\nConvBlock\n32ch · 256×256", DEC)
    arrow((x_mid + 1.6, y_levels[32] + 0.42), (x_right - 0.3, y_levels[64] - box_h / 2), color=DEC,
          label="up-conv /2", label_xy=(x_right - 1.0, (y_levels[64] + y_levels[32]) / 2 - 0.15))
    arrow((x_right, y_levels[64] - box_h / 2 + 0.95), (x_right, y_levels[128] - box_h / 2), color=DEC,
          label="up-conv /2", label_xy=(x_right + 1.35, (y_levels[128] + y_levels[64]) / 2))
    arrow((x_right, y_levels[128] - box_h / 2 + 0.95), (x_right, y_levels[256] - box_h / 2), color=DEC,
          label="up-conv /2", label_xy=(x_right + 1.35, (y_levels[256] + y_levels[128]) / 2))

    # --- Skip connections (dashed, horizontal) ---
    for res in (256, 128, 64):
        y = y_levels[res]
        arrow((x_left + box_w / 2, y), (x_right - box_w / 2, y), color=SKIP, lw=1.6, ls="--")
    ax.text((x_left + x_right) / 2, y_levels[256] + 0.62, "skip connections (concatenate)",
            fontsize=9, color=SKIP, ha="center")

    # --- Head + input/output ---
    head_y = y_levels[256]
    block(head_x, head_y, "head\n1×1 Conv\n3ch · 256×256", "#2ca02c", w=head_w)
    arrow((x_right + box_w / 2, head_y), (head_x - head_w / 2, head_y), color=DEC)

    ax.annotate("input\n3×256×256", xy=(x_left - box_w / 2, y_levels[256]),
                xytext=(x_left - 1.85, y_levels[256] + 1.05), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color="#333", lw=1))
    ax.annotate("per-pixel\nclass scores", xy=(head_x + head_w / 2, head_y),
                xytext=(head_x + head_w / 2 + 1.5, head_y - 1.05), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color="#333", lw=1))

    ax.legend(
        handles=[
            patches.Patch(facecolor=ENC, alpha=0.3, edgecolor=ENC, label="encoder (downsamples)"),
            patches.Patch(facecolor=BOTTLE, alpha=0.3, edgecolor=BOTTLE, label="bottleneck"),
            patches.Patch(facecolor=DEC, alpha=0.3, edgecolor=DEC, label="decoder (upsamples)"),
            plt.Line2D([0], [0], color=SKIP, lw=1.6, ls="--", label="skip connection"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9.5, frameon=False,
    )
    fig.suptitle("PhenoSegNet: a small U-Net", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "unet_schematic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_mask_example()
    figure_pixel_iou()
    figure_miou_good_bad()
    figure_unet_schematic()
    print(f"Wrote figures to {OUT_DIR}/")
