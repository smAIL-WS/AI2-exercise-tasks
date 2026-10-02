"""
Exercise 07 - Few-Shot Segmentation Dataset
===============================================
Same PhenoBench images/masks as Exercise 06. The only new piece is
restricting the training split to a small, seeded subset of `num_shots`
images -- the few-shot regime this exercise is about. Val/test loading is
unchanged from Exercise 06 (pass num_shots=None to use the whole split).

Run this script to verify:
    python src/dataset.py
"""

import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhenoBenchFewShotSegDataset(Dataset):
    """
    Semantic segmentation dataset for PhenoBench, with optional few-shot
    subsampling of the file list.

    Args:
        root_dir (str): Path to split folder (e.g. 'data/phenobench/train').
        transforms (callable, optional): Joint transforms for (image, mask).
        num_shots (int, optional): If set, deterministically sample this
            many images from the split instead of using all of them.
        seed (int): Seed for the few-shot sample, so the same subset is
            used every run.
    """

    def __init__(self, root_dir, transforms=None, num_shots=None, seed=42):
        self.root_dir = root_dir
        self.transforms = transforms

        # TODO: Set up image/mask directories and the (possibly few-shot)
        #       file list.
        #
        #   Step 1: Build paths to the images/ and masks/ subdirectories
        #           inside root_dir. Store as self.images_dir and
        #           self.masks_dir.
        #
        #   Step 2: List all image files in self.images_dir, sorted.
        #           Each image has a corresponding mask with the same
        #           filename in self.masks_dir.
        #
        #   Step 3: If num_shots is not None, deterministically sample
        #           num_shots filenames using a seeded random.Random(seed)
        #           instance -- do NOT use the global `random` module,
        #           since that would affect other code relying on global
        #           random state (e.g. shuffling in the DataLoader).
        #           Re-sort the sampled filenames afterwards so iteration
        #           order stays stable across runs.
        #
        #   Store the final list as self.filenames.

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
        #   Step 3: Load the mask with PIL. Do not convert to RGB -- it's
        #           a single-channel image where pixel values are class
        #           indices (0, 1, 2).
        #
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
    from transforms import get_train_transforms, get_val_transforms

    data_root = "data/phenobench"
    num_shots = 20

    train_dataset = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(data_root, "train"),
        transforms=get_train_transforms(512),
        num_shots=num_shots,
        seed=42,
    )
    val_dataset = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(data_root, "val"),
        transforms=get_val_transforms(512),
    )

    print(f"Few-shot train images: {len(train_dataset)} (requested {num_shots})")
    print(f"Full val images:       {len(val_dataset)}")

    image, mask = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Unique mask values: {torch.unique(mask).tolist()}")

    # Same seed -> same subset, every run.
    train_dataset_again = PhenoBenchFewShotSegDataset(
        root_dir=os.path.join(data_root, "train"),
        num_shots=num_shots,
        seed=42,
    )
    assert train_dataset.filenames == train_dataset_again.filenames, \
        "Few-shot subset is not deterministic -- check your seeding."
    print("\nSame seed reproduces the same few-shot subset.")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    images, masks = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")

    print("\n--- Dataset verified. ---")
