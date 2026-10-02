"""
Exercise 08 - LoRA Demo
=============================================
Wraps a pretrained SegFormer (the same checkpoint Ex07 used) with a LoRA
adapter via HuggingFace `peft`, and reports how many parameters that adds
-- for comparison against Ex07's linear-probe and partial-unfreeze
trainable-parameter counts.

Setup (once): pip install peft

Run this script to verify:
    python src/lora_demo.py            # default rank r=8 (this exercise's setting)
    python src/lora_demo.py --r 32     # try a different rank and watch the count move
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSemanticSegmentation


def load_base_segformer(model_name, num_classes):
    """
    Load a pretrained SegFormer with a replaced classification head.

    Identical to Ex07's load_pretrained_segformer (solutions/model.py) --
    LoRA wraps this same base model, it doesn't change how it's loaded.
    """
    return AutoModelForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


def apply_lora(model, r=8, lora_alpha=16, target_modules="all-linear", modules_to_save=("decode_head",)):
    """
    Wrap a frozen pretrained model with a trainable LoRA adapter.

    Args:
        model: a HuggingFace model (e.g. from load_base_segformer).
        r (int): LoRA rank -- the only real knob. Larger r means more
            trainable parameters and more expressive updates.
        lora_alpha (int): LoRA's scaling factor, conventionally set to
            2 * r (which is exactly the README's r=8, lora_alpha=16).
        target_modules (str): which layers get a LoRA adapter.
            "all-linear" finds every nn.Linear layer automatically, so
            this doesn't need to know SegFormer's internal layer names.
        modules_to_save (tuple[str]): modules to keep fully trainable
            *in addition to* the LoRA adapters. SegFormer's decode head
            classifier is a 1x1 Conv2d, not a Linear layer, so
            "all-linear" can't reach it -- without listing it here, the
            head stays frozen at its randomly re-initialized weights and
            the model could never actually learn PhenoBench's 3 classes.

    Returns:
        A peft-wrapped model. Everything not matched by target_modules
        or listed in modules_to_save is frozen automatically -- you
        don't need to freeze anything yourself here.
    """
    # TODO: Build a LoraConfig and wrap the model with it.
    #
    #   Step 1: Build a LoraConfig(r=r, lora_alpha=lora_alpha,
    #           target_modules=target_modules,
    #           modules_to_save=list(modules_to_save)).
    #
    #   Step 2: Wrap the model with get_peft_model(model, config) and
    #           return the result.
    pass


def count_trainable_parameters(model):
    """
    Manually count trainable vs. total parameters.

    Returns:
        dict with keys: trainable, total, trainable_pct
    """
    # TODO: Iterate model.named_parameters().
    #
    #   Step 1: Sum numel() across every parameter for "total".
    #   Step 2: Sum numel() again, but only for parameters with
    #           requires_grad == True, for "trainable".
    #   Step 3: trainable_pct = 100 * trainable / total.
    pass


def sample_trainable_parameter_names(model, n=8):
    """
    Return the names of the first n trainable parameters.

    Useful for spot-checking *where* LoRA added trainable weights --
    unlike Ex07's linear probe, these names should live inside
    "...segformer...", not just "decode_head...".
    """
    # TODO: List comprehension over model.named_parameters(), keeping
    #       only names where requires_grad is True, sliced to the
    #       first n.
    pass


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=8,
                         help="LoRA rank -- try 1, 8, 32 and watch the trainable count move.")
    args = parser.parse_args()

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_classes = 3

    base_model = load_base_segformer(model_name, num_classes)
    lora_model = apply_lora(base_model, r=args.r, lora_alpha=2 * args.r)

    print(f"LoRA rank r={args.r}, lora_alpha={2 * args.r}\n")
    lora_model.print_trainable_parameters()

    summary = count_trainable_parameters(lora_model)
    print(
        f"\nManual count -- trainable: {summary['trainable']:,} / "
        f"total: {summary['total']:,} ({summary['trainable_pct']:.3f}%)"
    )
    print("(should match peft's own count above)")

    names = sample_trainable_parameter_names(lora_model, n=8)
    print("\nFirst trainable parameter names:")
    for name in names:
        print(f"  {name}")
    print("\nNotice these live inside the encoder (e.g. '...segformer...lora_...'),")
    print("not just the head -- unlike Ex07's linear probe.")

    dummy_input = torch.randn(1, 3, 512, 512)
    lora_model.eval()
    with torch.no_grad():
        outputs = lora_model(pixel_values=dummy_input)
    print(f"\nForward pass still works -- logits shape: {list(outputs.logits.shape)}")

    print("\n--- LoRA model verified. Compare this trainable count against")
    print("your Ex07 linear-probe and partial-unfreeze numbers in worksheet.md. ---")
