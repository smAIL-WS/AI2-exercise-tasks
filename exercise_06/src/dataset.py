"""
Exercise 06 - Segmentation Dataset
=====================================
Returns (image, mask) pairs. Unlike detection, no custom collate_fn
needed — all masks have the same shape after resizing.

Run this script to verify:
    python src/dataset.py
"""

import os
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhenoBenchSegDataset(Dataset):
    """
    Semantic segmentation dataset for PhenoBench.

    Args:
        root_dir (str): Path to split folder (e.g. 'data/phenobench/train').
        transforms (callable, optional): Joint transforms for (image, mask).
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # TODO: Set up image/mask directories and file list.
        #
        #   Step 1: Build paths to the images/ and masks/ subdirectories
        #           inside root_dir. Store as self.images_dir and
        #           self.masks_dir.
        #
        #   Step 2: List all image files in self.images_dir, sorted.
        #           Each image has a corresponding mask with the same
        #           filename in self.masks_dir.
        #           Store as self.filenames (list of filename strings).

    def __len__(self):
        # TODO: Return the number of image-mask pairs.
        pass

    def __getitem__(self, idx):
        """
        Load one image-mask pair.

        Returns:
            image: Tensor (C, H, W) after transforms.
            mask: Long tensor (H, W) with class indices after transforms.
        """
        # TODO: Load and return one (image, mask) pair.
        #
        #   Step 1: Get the filename for this index.
        #
        #   Step 2: Load the image with PIL, convert to RGB.
        #
        #   Step 3: Load the mask with PIL. Do not convert to RGB — it's
        #           a single-channel image where pixel values are class
        #           indices (0, 1, 2).
        #
        #   Step 3b: Remap the mask labels according to LABEL_REMAP.
        #            We did this for you. It is necessary because the 
        #            original dataset has 5 classes, but we only want 3.
        
        # Remap labels
        remapped = np.zeros_like(mask)
        for src, dst in self.LABEL_REMAP.items():
            remapped[mask == src] = dst
        mask = Image.fromarray(remapped.astype(np.uint8))   
        
        #   Step 4: Apply self.transforms(image, mask) if transforms exist.
        #
        #   Step 5: Return (image, mask).
        pass


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_train_transforms

    data_root = "data/phenobench_256"

    dataset = PhenoBenchSegDataset(
        root_dir=os.path.join(data_root, "train"),
        transforms=get_train_transforms(256),
    )

    print(f"Total images: {len(dataset)}")

    image, mask = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Unique mask values: {torch.unique(mask).tolist()}")

    for cls_id, cls_name in enumerate(["background", "crop", "weed"]):
        count = (mask == cls_id).sum().item()
        total = mask.numel()
        print(f"  {cls_name}: {count} pixels ({100 * count / total:.1f}%)")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, masks = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")

    print("\n--- Dataset verified. ---")
