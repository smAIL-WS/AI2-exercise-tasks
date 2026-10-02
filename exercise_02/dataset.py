"""
Exercise 02 - Datasets and DataLoaders
=======================================
Solve each task below. Run this script to verify your solutions:
    python dataset.py

This script requires sample images in:
    data/sample_images/
    ├── class_a/
    │   ├── img1.jpg, img2.jpg, img3.jpg
    └── class_b/
        ├── img1.jpg, img2.jpg, img3.jpg
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exercise_02", "sample_images")

# ===========================================================================
# Task 1: Define a Transform Pipeline
# ===========================================================================
# Create a transforms.Compose() pipeline with three steps:
#   1. Resize images to (32, 32)
#   2. Convert PIL image to tensor using transforms.ToTensor()
#   3. Normalize with ImageNet mean=[0.485, 0.456, 0.406]
#      and std=[0.229, 0.224, 0.225]
#
# Store the result in a variable called 'transform'.
transform = None


# ===========================================================================
# Task 2: Build a Custom Dataset
# ===========================================================================
# Define a class SimpleImageDataset that subclasses torch.utils.data.Dataset.
class SimpleImageDataset(Dataset):
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
        #   Step 1: Create a mapping from class name to integer index.
        #           e.g. {"class_a": 0, "class_b": 1} (Hint: Use .enumerate() to get the index)
        #           Store as self.class_to_idx.
        self.class_to_idx = None
        #   Step 2: Build a list of (file_path, label) tuples.
        #           Loop over each class folder, then over each file inside it.
        #           Store the full path and the integer label.
        #           Store as self.samples.
        self.samples = None

    def __len__(self):
        #   Return the total number of samples.
        pass

    def __getitem__(self, idx):
        #   Step 1: Get the file path and label from self.samples[idx].
        #   Step 2: Open the image with PIL and convert to "RGB".
        #   Step 3: Apply self.transform if it is not None.
        #   Step 4: Return (image, label).
        pass

if __name__ == "__main__":
    # ===========================================================================
    # Task 3: Create a DataLoader
    # ===========================================================================
    # Step 1: Instantiate SimpleImageDataset with SAMPLE_DIR
    #         and the transform from Task 1.
    dataset = None
    dataloader = None
    #
    # Step 2: Print len(dataset) and dataset.classes.
    #
    # Step 3: Create a DataLoader with batch_size=2, shuffle=True, num_workers=0.
    #
    # Step 4: Fetch one batch using next(iter(dataloader)).
    #         Print the image tensor shape and the labels.
    
    
    # ===========================================================================
    # Task 4: Inspect a Single Sample
    # ===========================================================================
    # Step 1: Access the first sample directly: image, label = dataset[0]
    #
    # Step 2: Print the image tensor shape, its min and max pixel values,
    #         and the label.
    #
    # Step 3: The pixel values are no longer in [0, 1]. Why?
    #         Write your answer as a comment.
    #         Hint: think about what Normalize() does.
    
    
    # ===========================================================================
    # Verification
    # ===========================================================================
    print("\n=== VERIFICATION ===")

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


    def _scan_expected(sample_dir):
        if not os.path.isdir(sample_dir):
            return [], 0
        classes = sorted(
            d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))
        )
        total = sum(
            len([f for f in os.listdir(os.path.join(sample_dir, c))
                 if os.path.isfile(os.path.join(sample_dir, c, f))])
            for c in classes
        )
        return classes, total


    _expected_classes, _expected_samples = _scan_expected(SAMPLE_DIR)
    _expected_class_to_idx = {c: i for i, c in enumerate(_expected_classes)}

    results = []

    # A black pixel (0.0 after ToTensor) must become (0 - mean) / std per channel.
    _expected_black = torch.tensor([(0.0 - m) / s for m, s in
                                    zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    results.append(check(
        "Task 1 - transform resizes and tensor-ifies a sample image to (3, 32, 32)",
        lambda: _assert_true(
            transform(Image.new("RGB", (50, 50))).shape == torch.Size([3, 32, 32]),
            "expected transform(image) to produce a (3, 32, 32) tensor")))
    results.append(check(
        "Task 1 - transform normalizes with the ImageNet mean/std",
        lambda: torch.testing.assert_close(
            transform(Image.new("RGB", (50, 50), color=(0, 0, 0)))[:, 0, 0], _expected_black,
            msg="a black image should map to (0 - mean) / std per channel - check Normalize()")))

    results.append(check(
        "Task 2 - class_to_idx maps each class name to its sorted index",
        lambda: _assert_true(
            SimpleImageDataset(SAMPLE_DIR).class_to_idx == _expected_class_to_idx,
            f"expected {_expected_class_to_idx}")))
    results.append(check(
        "Task 2 - samples is a list of (file_path, label) tuples",
        lambda: _assert_true(
            all(isinstance(s, tuple) and len(s) == 2 and os.path.isfile(s[0]) and isinstance(s[1], int)
                for s in SimpleImageDataset(SAMPLE_DIR).samples),
            "expected every entry of self.samples to be (existing file path, int label)")))
    results.append(check(
        "Task 2 - SimpleImageDataset finds all sample images",
        lambda: _assert_true(
            len(SimpleImageDataset(SAMPLE_DIR)) == _expected_samples,
            f"expected {_expected_samples} samples under {SAMPLE_DIR}")))
    results.append(check(
        "Task 2 - SimpleImageDataset.__getitem__ returns (image, label)",
        lambda: _assert_true(
            (lambda sample: len(sample) == 2 and isinstance(sample[1], int))(SimpleImageDataset(SAMPLE_DIR)[0]),
            "expected dataset[idx] to return an (image, label) pair with an int label")))

    results.append(check("Task 3 - images shape is (2, 3, 32, 32)",
                          lambda: _assert_true(images.shape == torch.Size([2, 3, 32, 32]),
                                                f"expected shape (2, 3, 32, 32), got {tuple(images.shape)}")))
    results.append(check("Task 3 - labels has 2 entries",
                          lambda: _assert_true(len(labels) == 2, f"expected 2 labels, got {len(labels)}")))
    results.append(check(
        "Task 3 - dataset.classes matches the sample_images subfolders",
        lambda: _assert_true(dataset.classes == _expected_classes,
                              f"expected classes {_expected_classes}, got {dataset.classes}")))

    results.append(check("Task 4 - image shape is (3, 32, 32)",
                          lambda: _assert_true(image.shape == torch.Size([3, 32, 32]),
                                                f"expected shape (3, 32, 32), got {tuple(image.shape)}")))
    results.append(check(
        "Task 4 - label is a valid class index",
        lambda: _assert_true(0 <= label < len(dataset.classes),
                              f"expected label in [0, {len(dataset.classes)}), got {label}")))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
