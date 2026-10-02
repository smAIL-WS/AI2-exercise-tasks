"""
Exercise 03 - Transform Pipelines (SOLUTION)
==============================================
"""

from torchvision import transforms


def get_train_transforms(image_size, augmentation=True):
    if augmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transform


def get_val_transforms(image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform


if __name__ == "__main__":
    import torch

    image_size = 128

    train_tf = get_train_transforms(image_size, augmentation=True)
    val_tf = get_val_transforms(image_size)

    print("Train transforms (with augmentation):")
    print(train_tf)
    print(f"\nVal transforms:")
    print(val_tf)

    from PIL import Image
    dummy_img = Image.new("RGB", (256, 256), color=(100, 150, 200))

    train_out = train_tf(dummy_img)
    val_out = val_tf(dummy_img)

    print(f"\nTrain output shape: {train_out.shape}")
    print(f"Val output shape: {val_out.shape}")
    print("\n--- Transforms verified. ---")
