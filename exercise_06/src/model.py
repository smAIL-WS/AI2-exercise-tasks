"""
Exercise 06 - Segmentation Model Factory
===========================================
CNN and transformer-based segmentation models, all sharing the same API:
    Input:  (B, 3, H, W)
    Output: (B, num_classes, H, W)

Swap models by changing model.name in config.yaml.

CNN models (torchvision):
    "unet", "deeplabv3_resnet50", "fcn_resnet50"

Transformer models (HuggingFace):
    "segformer_b0" — smallest SegFormer, fast, ~3.7M params
    "segformer_b2" — mid-size SegFormer, better accuracy, ~27M params

Note: torchvision models (DeepLabV3, FCN) return a dict {"out": tensor}.
U-Net and SegFormer return a plain tensor. Handle in train/eval:
    if isinstance(output, dict): output = output["out"]

Run this script to verify:
    python src/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models


class ConvBlock(nn.Module):
    """Two 3x3 Conv-BN-ReLU layers. Given fully implemented."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Architecture:
        Encoder: 4 blocks (3→64→128→256→512) with MaxPool2d between
        Bottleneck: 512→1024
        Decoder: 4 blocks with ConvTranspose2d + skip connections
        Head: 1×1 Conv → num_classes
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # --- Encoder ---
        # TODO: Define four encoder ConvBlocks and a shared MaxPool2d(2).
        #
        #   Channel progression: 3→64, 64→128, 128→256, 256→512.
        #   The pool is shared (no learnable params).

        # --- Bottleneck ---
        # TODO: Define the bottleneck ConvBlock: 512→1024.

        # --- Decoder ---
        # TODO: Define four decoder stages. Each stage has:
        #   a) ConvTranspose2d — doubles spatial size, halves channels. 
        #      Use kernel_size=2 and stride=2.
        #   b) ConvBlock — processes concatenated (upsampled + skip) features.
        #
        #   The ConvBlock input is DOUBLE the up-conv output because of
        #   the skip concatenation. For example: up1 outputs 512 channels,
        #   concatenated with 512 from enc4 = 1024 input to dec1.
        #
        #   Four stages: (1024→512, 1024→512), (512→256, 512→256),
        #                (256→128, 256→128), (128→64, 128→64)

        # --- Head ---
        # TODO: 1×1 Conv2d from 64 channels to num_classes.

        pass

    def forward(self, x):
        """
        Forward pass with skip connections.
        Returns (B, num_classes, H, W).
        """
        # TODO: Implement the U-Net forward pass.
        #
        #   Encoder: run each ConvBlock and save its output BEFORE pooling.
        #   These saved tensors are the skip connections.
        #       e1 from enc1, then pool
        #       e2 from enc2, then pool
        #       e3 from enc3, then pool
        #       e4 from enc4, then pool
        #
        #   Bottleneck: run on the pooled output of enc4.
        #
        #   Decoder: for each stage, upsample with ConvTranspose2d,
        #   concatenate with the MATCHING skip along dim=1 (hint: torch.cat([d, e], dim=1)), then ConvBlock.
        #   Skips pair in REVERSE order: e4 with first decoder stage,
        #   e1 with last — because the decoder climbs back up in resolution.
        #
        #   Head: return self.final_conv(last_decoder_output).

        pass


# ---------------------------------------------------------------------------
# SegFormer wrapper (given) — adapts HuggingFace API to exercise convention
# ---------------------------------------------------------------------------
class SegFormerWrapper(nn.Module):
    """
    Wraps HuggingFace SegFormer so it returns (B, num_classes, H, W) at
    full input resolution — same as U-Net and torchvision models.

    SegFormer natively outputs logits at 1/4 of input resolution. This
    wrapper upsamples them internally so train.py and evaluate.py don't
    need special handling.

    Given fully implemented — no TODOs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        # Upsample to match input resolution
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_segmentation_model(name, num_classes, pretrained=True):
    """
    Factory function for segmentation models.

    All models return (B, num_classes, H, W) tensors directly,
    EXCEPT torchvision models (DeepLabV3, FCN) which return a dict.
    Handle with: if isinstance(output, dict): output = output["out"]

    Args:
        name (str): Model identifier from config.yaml.
        num_classes (int): Number of output classes.
        pretrained (bool): Load pretrained weights (ignored for unet).

    Returns:
        nn.Module: Segmentation model.
    """
    name = name.lower()

    if name == "unet":
        model = UNet(num_classes=num_classes)

    elif name == "deeplabv3_resnet50":
        # TODO: Load DeepLabV3 from torchvision and replace its head.
        #
        #   Step 1: Load with seg_models.deeplabv3_resnet50().
        #           Use weights="DEFAULT" if pretrained, else None.
        #
        #   Step 2: The classification head is at model.classifier[-1].
        #           Read its in_channels, then replace it with a new
        #           Conv2d(in_channels, num_classes, kernel_size=1).
        #
        #   Step 3: Replace the auxiliary classifier head IF it exists.
        #           model.aux_classifier is None when pretrained=False.
        #           Check with: if model.aux_classifier is not None
        model = None

    elif name == "fcn_resnet50":
        # TODO: Same pattern as DeepLabV3 using seg_models.fcn_resnet50().
        model = None

    elif name in ("segformer_b0", "segformer_b2"):
        # TODO: Load SegFormer from HuggingFace and wrap it.
        #
        #   Step 1: Import SegformerForSemanticSegmentation from the
        #           transformers library.
        #
        #   Step 2: Choose the pretrained checkpoint based on the name (create a dict).
        #           B0 uses "nvidia/segformer-b0-finetuned-ade-512-512"
        #           B2 uses "nvidia/segformer-b2-finetuned-ade-512-512"
        checkpoints = {} # TODO: fill in the dict with the two checkpoints

        #   Step 3: Load with .from_pretrained() if pretrained. Otherwise,
        #           load a SegformerConfig from the same checkpoint and
        #           set config.num_labels to num_classes, then create
        #           the model from that config.
        if pretrained:
            hf_model = None # TODO: load pretrained model
        else:
            from transformers import SegformerConfig
            # TODO: load the config from the checkpoint, set num_labels, then create the model
        
        #   Step 4: The pretrained model has 150 ADE20K classes. Replace
        #           the decode head classifier for num_classes. It lives at
        #           model.decode_head.classifier — a Conv2d layer. Read its
        #           in_channels and replace with a new Conv2d(in_channels,
        #           num_classes, kernel_size=1).
        #
        #   Step 5: Wrap with SegFormerWrapper (given above) so the output
        #           is automatically upsampled to full resolution.
        #           Return the wrapper, not the raw HuggingFace model.
        model = None

    else:
        raise ValueError(
            f"Unknown model: '{name}'. Choose from: "
            f"unet, deeplabv3_resnet50, fcn_resnet50, segformer_b0, segformer_b2"
        )

    return model


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    num_classes = 3
    dummy_input = torch.randn(2, 3, 256, 256)

    model_names = ["unet", "deeplabv3_resnet50", "fcn_resnet50",
                   "segformer_b0", "segformer_b2"]

    print(f"{'Model':<22} {'Params':>12} {'Output Shape':>18} {'Type':<15}")
    print("-" * 70)

    for name in model_names:
        model = get_segmentation_model(name, num_classes=num_classes, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                output = output["out"]

        params = count_parameters(model)
        arch_type = "transformer" if "segformer" in name else "CNN"
        print(f"{name:<22} {params:>12,} {str(list(output.shape)):>18} {arch_type:<15}")

        assert output.shape[2:] == dummy_input.shape[2:], \
            f"{name}: output dims {output.shape[2:]} != input {dummy_input.shape[2:]}"

    print("\nAll models (CNN + transformer) output (B, num_classes, H, W) at full resolution.")
    print("\n--- Model verified. ---")
