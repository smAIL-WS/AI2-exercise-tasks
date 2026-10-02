"""
Exercise 03 - Custom Dataset (SOLUTION)
=========================================
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PlantWildDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            label = self.class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                filepath = os.path.join(cls_dir, fname)
                if os.path.isfile(filepath):
                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_train_transforms, get_val_transforms

    image_size = 128
    data_root = "data/plantwild"

    train_dataset = PlantWildDataset(
        root_dir=os.path.join(data_root, "train"),
        transform=get_train_transforms(image_size, augmentation=True),
    )
    val_dataset = PlantWildDataset(
        root_dir=os.path.join(data_root, "val"),
        transform=get_val_transforms(image_size),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Class-to-index mapping: {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    images, labels = next(iter(train_loader))

    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch labels: {labels}")
    print(f"Label dtype: {labels.dtype}")

    from collections import Counter
    label_counts = Counter([s[1] for s in train_dataset.samples])
    print("\nPer-class sample counts (train):")
    for cls_name in train_dataset.classes:
        idx = train_dataset.class_to_idx[cls_name]
        print(f"  {cls_name}: {label_counts[idx]}")

    print("\n--- Dataset verified. ---")
