"""
Exercise 04 - Model Factory
==============================
Provides both the custom PlantCNN from Exercise 03 and pretrained SOTA
architectures from torchvision. The get_model() factory function lets
you switch between them by changing a single field in config.yaml.

Why use pretrained models?
    Architectures like ResNet, VGG, and AlexNet were designed and optimized
    by research teams over years of experimentation. Their layer depths,
    channel widths, and skip connections are already validated on millions
    of images (ImageNet). Building your own CNN from scratch (like PlantCNN)
    is valuable for learning, but in practice you almost always start from
    a pretrained architecture and fine-tune it for your specific dataset.
    This is called transfer learning.

Supported models:
    - "plantcnn"  : Custom CNN from Exercise 03 (baseline)
    - "resnet18"  : 11.7M params — best balance of speed and accuracy (recommended)
    - "resnet50"  : 25.6M params — deeper, more capacity, slower
    - "vgg16"     : 138M params  — historically important, very large
    - "alexnet"   :  61M params  — first deep CNN to win ImageNet (2012), shallow

Run this script to compare all models:
    python src/model.py
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Custom CNN from Exercise 03
# ---------------------------------------------------------------------------
class PlantCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_model(name, num_classes, pretrained=True, dropout=0.5):
    """
    Factory function to create a classification model.

    For pretrained models, the final classification layer is replaced to
    match the number of classes in the dataset. All other layers retain
    their ImageNet-pretrained weights — this is transfer learning.

    Args:
        name (str): Model name. One of:
            "plantcnn", "resnet18", "resnet50", "vgg16", "alexnet"
        num_classes (int): Number of output classes.
        pretrained (bool): If True, load ImageNet-pretrained weights.
            Ignored for "plantcnn" (no pretrained weights available).
        dropout (float): Dropout probability. Used by PlantCNN and VGG.

    Returns:
        nn.Module: The model ready for training.
    """
    name = name.lower()

    if name == "plantcnn":
        # Custom CNN from Exercise 03 — no pretrained weights
        model = PlantCNN(num_classes=num_classes, dropout=dropout)

    elif name == "resnet18":
        # ResNet-18: 11.7M params
        # Replace the final fully connected layer (originally 1000 ImageNet classes)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        # ResNet-50: 25.6M params
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vgg16":
        # VGG-16: 138M params — very large, may overfit on small datasets
        # Replace the last layer in the classifier sequential block
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    elif name == "alexnet":
        # AlexNet: 61M params — the original deep CNN (Krizhevsky et al., 2012)
        # Historically important but shallow by modern standards
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Choose from: plantcnn, resnet18, resnet50, vgg16, alexnet"
        )

    return model


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Verification — compare all models
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_classes = 10
    dummy_input = torch.randn(2, 3, 128, 128)

    model_names = ["plantcnn", "resnet18", "resnet50", "vgg16", "alexnet"]

    print(f"{'Model':<12} {'Params':>12} {'Output Shape':>15}")
    print("-" * 42)

    for name in model_names:
        model = get_model(name, num_classes=num_classes, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        params = count_parameters(model)
        print(f"{name:<12} {params:>12,} {str(list(output.shape)):>15}")

    print("\nRecommended: resnet18 (best speed/accuracy trade-off)")
    print("Try changing model.name in config.yaml and compare results.")
