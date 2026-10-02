"""
Exercise 05 - Detection Dataset
==================================
Custom Dataset for object detection using COCO JSON annotations.

Unlike classification where __getitem__ returns (image, label),
detection returns (image, target_dict) where target contains
bounding boxes, labels, areas, and image IDs for all objects.

Run this script to verify:
    python src/dataset.py
"""

import os
import json
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhenoBenchDetectionDataset(Dataset):
    """
    Detection dataset for PhenoBench using COCO JSON annotations.

    Args:
        root_dir (str): Path to split folder (e.g. 'data/phenobench/train').
        annotation_file (str): Path to COCO JSON file.
        transforms (callable, optional): Detection transforms operating
            on (image, target) pairs.
    """

    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # TODO: Load and parse the COCO JSON annotation file.
        #
        #   Step 1: Open annotation_file and parse it with json.load().
        #           The JSON contains three keys: "images", "annotations",
        #           "categories".
        #
        #   Step 2: Store the image list as self.images and the category
        #           list as self.categories.
        self.images = None
        self.categories = None
        #
        #   Step 3: Build a lookup from image_id to its annotations.
        #           Create a defaultdict(list) called self.img_to_anns.
        #           Loop over all annotation dicts — each has an "image_id"
        #           field. Append each annotation to the list for its
        #           image_id. This way, self.img_to_anns[some_id] gives
        #           you all objects in that image.

    def __len__(self):
        # TODO: Return the number of images in the dataset.
        #       Hint: self.images is a list.
        pass

    def __getitem__(self, idx):
        """
        Load one image and its detection targets.

        Returns:
            image: Image tensor.
            target (dict): Contains boxes, labels, image_id, area, iscrowd.
        """
        # TODO: Load an image and build its target dictionary.
        #
        #   Step 1: Get the image info dict from self.images[idx].
        #           Extract "id" (image_id) and "file_name".
        img_info = None
        image_id = None
        file_name = None
        #
        #   Step 2: Build the full path to the image file:
        #           os.path.join(self.root_dir, "images", file_name)
        #           Open it with PIL and convert to "RGB".
        img_path = None
        image = None
        #
        #   Step 3: Get all annotations for this image using
        #           self.img_to_anns[image_id].
        annotations = None
        #
        #   Step 4: Loop over the annotations. For each one:
        #           - Extract the bbox field: [x, y, w, h] (COCO format)
        #           - Convert to [x1, y1, x2, y2]: x1=x, y1=y, x2=x+w, y2=y+h
        #           - Collect the category_id as the label
        #           - Collect the area value
        #           - Collect iscrowd (default to 0 if missing)
        #           Append each to separate lists (boxes, labels, areas, iscrowd).
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in annotations:
            
        
        #   Step 5: Convert lists to torch tensors and build the target dict:
        #           - "boxes":    FloatTensor of shape [N, 4]
        #           - "labels":   Int64Tensor of shape [N]
        #           - "image_id": Int64Tensor of shape [1]
        #           - "area":     FloatTensor of shape [N]
        #           - "iscrowd":  UInt8Tensor of shape [N]
        #           If no annotations exist for this image, create empty
        #           tensors with the correct shapes (e.g. torch.zeros((0, 4))
        #           for boxes).
        #
        #   Step 6: If self.transforms is not None, apply them:
        #           image, target = self.transforms(image, target)
        #
        #   Step 7: Return (image, target).
         pass


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_train_transforms
    from utils import collate_fn

    data_root = "data/phenobench_256"

    dataset = PhenoBenchDetectionDataset(
        root_dir=os.path.join(data_root, "train"),
        annotation_file=os.path.join(data_root, "annotations", "train.json"),
        transforms=get_train_transforms(),
    )

    print(f"Total images: {len(dataset)}")
    print(f"Categories: {dataset.categories}")

    image, target = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Number of objects: {len(target['boxes'])}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Labels: {target['labels']}")
    print(f"  Image ID: {target['image_id']}")

    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    images, targets = next(iter(loader))

    print(f"\nBatch:")
    print(f"  Number of images: {len(images)}")
    print(f"  Image 0 shape: {images[0].shape}")
    print(f"  Targets 0 objects: {len(targets[0]['boxes'])}")
    print(f"  Targets 1 objects: {len(targets[1]['boxes'])}")

    print("\n--- Dataset verified. ---")
