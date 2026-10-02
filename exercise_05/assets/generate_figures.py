#!/usr/bin/env python3
"""
One-off script that generates the illustrative figures embedded in
exercise_05/README.md (bounding box formats, IoU examples, mAP TP/FP
example). Not part of the graded exercise -- just documentation tooling.
Regenerate with:

    python assets/generate_figures.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path(__file__).parent

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#9467BD"
GRAY = "#555555"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def draw_box(ax, xyxy, color, linestyle="-", linewidth=2.5, fill_alpha=0.0, zorder=3):
    xmin, ymin, xmax, ymax = xyxy
    if fill_alpha > 0.0:
        fill = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=color, edgecolor="none", alpha=fill_alpha, zorder=zorder,
        )
        ax.add_patch(fill)
    edge = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=linewidth, edgecolor=color, facecolor="none",
        linestyle=linestyle, zorder=zorder + 1,
    )
    ax.add_patch(edge)


def dim_arrow(ax, xy_start, xy_end, text, color=GRAY, text_xy=None, fontsize=11):
    ax.annotate(
        "", xy=xy_end, xytext=xy_start,
        arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.5),
    )
    if text_xy is None:
        text_xy = ((xy_start[0] + xy_end[0]) / 2, (xy_start[1] + xy_end[1]) / 2)
    ax.text(text_xy[0], text_xy[1], text, color=color, fontsize=fontsize,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def point_label(ax, xy, text, xytext, color, fontsize=12, ha="center"):
    ax.plot(*xy, "o", color=color, ms=7, zorder=4)
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=color, ha=ha,
                va="center", arrowprops=dict(arrowstyle="-", color=color, lw=1))


def setup_axes(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)  # pass inverted (high, low) for image-coordinate convention
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(labelsize=9)


# ===========================================================================
# Figure 1: Bounding box formats (XYXY / XYWH / normalized cxcywh)
# ===========================================================================
def figure_bbox_formats():
    IMG = 640
    xmin, ymin, xmax, ymax = 180, 140, 300, 220
    w, h = xmax - xmin, ymax - ymin
    cx, cy = xmin + w / 2, ymin + h / 2

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    # --- Panel 1: Corners (XYXY) ---
    ax = axes[0]
    setup_axes(ax, (60, 400), (330, 60), "Corners — XYXY  (torchvision)")
    draw_box(ax, (xmin, ymin, xmax, ymax), BLUE, fill_alpha=0.12)
    point_label(ax, (xmin, ymin), f"(xmin, ymin) = ({xmin}, {ymin})",
                xytext=(70, 90), color=BLUE, ha="left")
    point_label(ax, (xmax, ymax), f"(xmax, ymax) = ({xmax}, {ymax})",
                xytext=(390, 300), color=BLUE, ha="right")
    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")

    # --- Panel 2: Corner + size (XYWH, COCO) ---
    ax = axes[1]
    setup_axes(ax, (60, 400), (330, 60), "Corner + size — XYWH  (COCO)")
    draw_box(ax, (xmin, ymin, xmax, ymax), ORANGE, fill_alpha=0.12)
    point_label(ax, (xmin, ymin), f"(x, y) = ({xmin}, {ymin})",
                xytext=(70, 175), color=ORANGE, ha="left")
    dim_arrow(ax, (xmin, ymin - 20), (xmax, ymin - 20), f"width = {w}",
              color=ORANGE, text_xy=(cx, 100))
    dim_arrow(ax, (xmax + 25, ymin), (xmax + 25, ymax), f"height = {h}",
              color=ORANGE, text_xy=(370, cy))
    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")

    # --- Panel 3: Center + size, normalized (YOLO) ---
    ax = axes[2]
    nxmin, nymin, nxmax, nymax = xmin / IMG, ymin / IMG, xmax / IMG, ymax / IMG
    ncx, ncy, nw, nh = cx / IMG, cy / IMG, w / IMG, h / IMG
    setup_axes(ax, (0.09, 0.62), (0.51, 0.09), "Center + size, normalized (YOLO)")
    draw_box(ax, (nxmin, nymin, nxmax, nymax), GREEN, fill_alpha=0.12)
    point_label(ax, (ncx, ncy), f"(cx, cy) = ({ncx:.3f}, {ncy:.3f})",
                xytext=(ncx, 0.14), color=GREEN)
    dim_arrow(ax, (nxmin, nymax + 0.035), (nxmax, nymax + 0.035), f"w = {nw:.3f}",
              color=GREEN, text_xy=(ncx, nymax + 0.07))
    dim_arrow(ax, (nxmax + 0.04, nymin), (nxmax + 0.04, nymax), f"h = {nh:.3f}",
              color=GREEN, text_xy=(0.58, ncy))
    ax.set_xlabel("x / image_width"); ax.set_ylabel("y / image_height")

    fig.suptitle("The same box, three ways", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bbox_formats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Figure 2: IoU examples (high / borderline / low overlap)
# ===========================================================================
def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union


def figure_iou_examples():
    # Boxes are centered in the panel (rather than near a corner) so they
    # don't run into the legend, which sits in the upper-left of panel 1.
    box_a = (105, 105, 205, 205)  # fixed 100x100 reference box

    cases = [
        ("High overlap", (110, 110, 210, 210)),
        ("Right at the threshold", (138, 105, 238, 205)),
        ("Low overlap", (165, 165, 265, 265)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))
    for ax, (title, box_b) in zip(axes, cases):
        iou = compute_iou(box_a, box_b)
        verdict = "match  ✓" if iou >= 0.5 else "no match  ✗"
        setup_axes(ax, (30, 290), (290, 30), title)

        ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        if ix2 > ix1 and iy2 > iy1:
            ax.add_patch(patches.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1,
                                            facecolor=PURPLE, alpha=0.45, zorder=2))

        draw_box(ax, box_a, BLUE)
        draw_box(ax, box_b, ORANGE)

        color = GREEN if iou >= 0.5 else RED
        ax.text(160, 275, f"IoU = {iou:.2f}  —  {verdict}",
                fontsize=12.5, color=color, ha="center", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(
        handles=[
            patches.Patch(edgecolor=BLUE, facecolor="none", linewidth=2.5, label="box A"),
            patches.Patch(edgecolor=ORANGE, facecolor="none", linewidth=2.5, label="box B"),
            patches.Patch(facecolor=PURPLE, alpha=0.45, label="intersection"),
        ],
        loc="upper left", fontsize=9, framealpha=0.9,
    )
    fig.suptitle("Jaccard overlap (IoU) at different amounts of overlap", fontsize=16, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iou_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Figure 3: mAP -- true positive vs. false positive
# ===========================================================================
def figure_map_tp_fp():
    gt = (80, 70, 190, 190)
    pred_tp = (88, 78, 198, 198)   # well-localized -> high IoU
    pred_fp = (165, 40, 255, 130)  # spurious, only grazes the ground truth -> low IoU

    iou_tp = compute_iou(gt, pred_tp)
    iou_fp = compute_iou(gt, pred_fp)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    setup_axes(ax, (55, 340), (280, 20),
               "IoU $\\geq$ 0.5 with an unclaimed ground-truth box = match")

    draw_box(ax, gt, GRAY, linestyle="--", linewidth=2.5)
    draw_box(ax, pred_tp, GREEN, fill_alpha=0.12)
    draw_box(ax, pred_fp, RED, fill_alpha=0.12)

    ax.text(gt[0], gt[1] - 8, "ground truth", color=GRAY, fontsize=11, va="bottom")
    ax.annotate(f"prediction A — score 0.91\nIoU {iou_tp:.2f} → True Positive",
                xy=((pred_tp[0] + pred_tp[2]) / 2, pred_tp[3]),
                xytext=(120, 250), fontsize=11, color=GREEN, va="top", ha="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
    ax.annotate(f"prediction B — score 0.77\nIoU {iou_fp:.2f} → False Positive",
                xy=(pred_fp[2] - 10, pred_fp[3]),
                xytext=(200, 220), fontsize=11, color=RED, va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "map_tp_fp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_bbox_formats()
    figure_iou_examples()
    figure_map_tp_fp()
    print(f"Wrote figures to {OUT_DIR}/")
