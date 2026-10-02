"""
Exercise 03 - Custom Dataset
==============================
Custom Dataset class for loading plant disease images from an ImageFolder
structure. Each subfolder name is a class label.

Run this script to verify:
    python src/dataset.py
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PlantWildDataset(Dataset):
    """
    Custom Dataset for plant disease classification.

    Expected folder structure:
        root_dir/
        ├── class_name_1/
        │   ├── img1.jpg
        │   └── ...
        └── class_name_2/
            └── ...

    Args:
        root_dir (str): Path to the split folder (e.g. 'data/plantwild/train').
        transform (callable, optional): Transform pipeline to apply to images.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Listing all subdirectories inside root_dir (ignore files).
        # Sorting them alphabetically — this defines the class order.
        # Storing as self.classes (list of strings)
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # TODO: class-to-index mapping, and samples list.
        #
        #
        #   Step 1: Create a dictionary mapping each class name to its
        #           integer index: {"apple_scab": 0, "tomato_blight": 1, ...}
        #           The indices follow the sorted order from Step 1.
        #           Store as self.class_to_idx.
        self.class_to_idx = None
        #
        #   Step 2: Build a list of (file_path, label) tuples.
        #           For each class folder, loop over its files. For each file,
        #           store the full path and the integer label from Step 1.
        #           Store as self.samples.
        self.samples = None
        

    def __len__(self):
        # TODO: Return the total number of samples.
        pass

    def __getitem__(self, idx):
        """
        Load and return one sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, label)
        """
        # TODO: Load one image and return it with its label.
        #
        #   Step 1: Get the file path and label from self.samples[idx].
        #   Step 2: Open the image with PIL and convert to "RGB".
        #           Some images may be grayscale or RGBA — converting
        #           to RGB ensures a consistent 3-channel input.
        #   Step 3: Apply self.transform if it is not None.
        #   Step 4: Return (image, label).
        pass


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import sys
    from collections import Counter
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import get_train_transforms, get_val_transforms

    print("=== VERIFICATION ===")

    def _assert_true(condition, message):
        assert condition, message


    def check(label, assertion_fn):
        try:
            assertion_fn()
        except AssertionError as e:
            print(f"[FAIL] {label} - {e}")
            return False
        except NameError as e:
            print(f"[FAIL] {label} (variable not defined: {e})")
            return False
        except Exception as e:
            print(f"[FAIL] {label} (error while checking: {e})")
            return False
        print(f"[PASS] {label}")
        return True


    def _scan_expected(split_dir):
        """Ground truth straight from the folder structure: sorted classes and files per class."""
        classes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
        files = {c: [f for f in os.listdir(os.path.join(split_dir, c))
                     if os.path.isfile(os.path.join(split_dir, c, f))] for c in classes}
        return classes, files


    image_size = 128
    data_root = "data/plantwild"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print(f"[SKIP] all dataset checks - '{train_dir}' / '{val_dir}' not found. "
              f"Run this script from the exercise folder (exercise_03/) after setting up the data (see README).")
        sys.exit(0)

    expected_classes, expected_files = _scan_expected(train_dir)
    expected_class_to_idx = {c: i for i, c in enumerate(expected_classes)}
    expected_num_samples = sum(len(f) for f in expected_files.values())


    def _check_class_to_idx():
        class_to_idx = PlantWildDataset(train_dir).class_to_idx
        _assert_true(class_to_idx is not None, "self.class_to_idx is still None")
        _assert_true(class_to_idx == expected_class_to_idx,
                     f"expected {{class_name: index}} following the sorted class order, "
                     f"e.g. {{'{expected_classes[0]}': 0, '{expected_classes[1]}': 1, ...}}")


    def _check_samples():
        samples = PlantWildDataset(train_dir).samples
        _assert_true(samples is not None, "self.samples is still None")
        _assert_true(all(isinstance(s, tuple) and len(s) == 2 for s in samples),
                     "expected every entry of self.samples to be a (file_path, label) tuple")
        _assert_true(len(samples) == expected_num_samples,
                     f"expected {expected_num_samples} samples (every file in every class folder), got {len(samples)}")
        for path, label in samples:
            _assert_true(os.path.isfile(path), f"'{path}' is not a file - store the full path to the image")
            class_name = os.path.basename(os.path.dirname(path))
            _assert_true(isinstance(label, int) and label == expected_class_to_idx.get(class_name),
                         f"'{path}' has label {label!r}, expected {expected_class_to_idx.get(class_name)} "
                         f"(the index of '{class_name}')")


    def _check_len():
        n = PlantWildDataset(train_dir).__len__()
        _assert_true(n is not None, "__len__ returned None - return the number of samples")
        _assert_true(n == expected_num_samples, f"expected len(dataset) == {expected_num_samples}, got {n}")


    def _check_getitem_raw():
        sample = PlantWildDataset(train_dir)[0]
        _assert_true(isinstance(sample, tuple) and len(sample) == 2,
                     "expected dataset[idx] to return an (image, label) tuple")
        image, label = sample
        _assert_true(isinstance(image, Image.Image) and image.mode == "RGB",
                     f"without a transform, expected a PIL image in RGB mode, got "
                     f"{getattr(image, 'mode', type(image).__name__)} - did you .convert('RGB')?")
        _assert_true(isinstance(label, int), f"expected an int label, got {type(label).__name__}")


    def _check_getitem_transformed():
        sample = PlantWildDataset(train_dir, transform=get_val_transforms(image_size))[0]
        _assert_true(sample is not None, "__getitem__ returned None - return (image, label)")
        image, label = sample
        _assert_true(isinstance(image, torch.Tensor) and image.shape == torch.Size([3, image_size, image_size]),
                     f"expected a (3, {image_size}, {image_size}) tensor - is self.transform applied?")


    def _check_batch():
        train_dataset = PlantWildDataset(train_dir, transform=get_train_transforms(image_size, augmentation=True))
        images, labels = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)))
        _assert_true(images.shape == torch.Size([4, 3, image_size, image_size]),
                     f"expected images of shape (4, 3, {image_size}, {image_size}), got {tuple(images.shape)}")
        _assert_true(labels.shape == torch.Size([4]) and labels.dtype == torch.int64,
                     f"expected 4 int64 labels, got shape {tuple(labels.shape)} with dtype {labels.dtype}")


    def _check_val_split():
        val_dataset = PlantWildDataset(val_dir, transform=get_val_transforms(image_size))
        _assert_true(len(val_dataset) > 0, "the val split is empty")
        _assert_true(val_dataset.class_to_idx == expected_class_to_idx,
                     "train and val must share the same class-to-index mapping")


    results = []

    results.append(check("PlantWildDataset - class_to_idx maps each class name to its sorted index",
                         _check_class_to_idx))
    results.append(check("PlantWildDataset - samples holds a (full file path, class index) tuple for every image",
                         _check_samples))
    results.append(check(f"PlantWildDataset - __len__ returns the number of samples ({expected_num_samples})",
                         _check_len))
    results.append(check("PlantWildDataset - __getitem__ returns (RGB image, int label)", _check_getitem_raw))
    results.append(check("PlantWildDataset - __getitem__ applies self.transform", _check_getitem_transformed))
    results.append(check("DataLoader - a batch has images (4, 3, 128, 128) and 4 int64 labels", _check_batch))
    results.append(check("PlantWildDataset - val split loads with the same class mapping as train",
                         _check_val_split))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")

    if all(results):
        label_counts = Counter(label for _, label in PlantWildDataset(train_dir).samples)
        print(f"\nPer-class sample counts (train, {len(expected_classes)} classes):")
        for cls_name in expected_classes:
            print(f"  {cls_name}: {label_counts[expected_class_to_idx[cls_name]]}")
