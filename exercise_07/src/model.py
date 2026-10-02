"""
Exercise 07 - Few-Shot Segmentation Model
=============================================
Loads a pretrained SegFormer from HuggingFace, replaces its head for
PhenoBench's 3 classes, and freezes the encoder so only the (small)
decode head trains -- the core few-shot finetuning strategy this
exercise is built around.

Run this script to verify:
    python src/model.py            # nothing frozen (default)
    python src/model.py --freeze   # encoder frozen, only the head trains
"""

import torch
from transformers import AutoModelForSemanticSegmentation


def load_pretrained_segformer(model_name, num_classes):
    """
    Load a pretrained SegFormer and replace its classification head.

    The pretrained checkpoint's decode head outputs its original training
    set's class count (e.g. 150 for ADE20K) -- ignore_mismatched_sizes=True
    lets it load anyway and re-initializes just the mismatched head layers
    for num_classes.

    Args:
        model_name (str): HuggingFace model id, e.g.
            "nvidia/segformer-b0-finetuned-ade-512-512".
        num_classes (int): Number of output classes (3 for PhenoBench).

    Returns:
        SegformerForSemanticSegmentation
    """
    # TODO: Load the model.
    #
    #   Use AutoModelForSemanticSegmentation.from_pretrained() with
    #   num_labels=num_classes and ignore_mismatched_sizes=True.
    pass


def freeze_encoder(model):
    """
    Freeze every encoder parameter, leave the decode head trainable.

    This is the required few-shot strategy: with only a handful of
    training images, the small decode head can plausibly learn something;
    the large, general-purpose encoder cannot be safely retrained on so
    little data without destroying what it already knows.

    Args:
        model (SegformerForSemanticSegmentation)

    Returns:
        The same model, with requires_grad set appropriately.
    """
    # TODO: Freeze the encoder, keep the head trainable.
    #
    #   Step 1: Set requires_grad = False for every parameter in
    #           model.segformer (the encoder).
    #
    #   Step 2: Set requires_grad = True for every parameter in
    #           model.decode_head (should already be True by default,
    #           but set it explicitly so the function is self-contained).
    return model


def parameter_summary(model):
    """
    Count trainable/total parameters, split into encoder vs. head.

    Returns:
        dict with keys: encoder_total, encoder_trainable,
        head_total, head_trainable, total, trainable.
    """
    # TODO: Build the summary dict.
    #
    #   Step 1: Iterate model.named_parameters(). A parameter belongs to
    #           the encoder if its name starts with "segformer.", and to
    #           the head if its name starts with "decode_head.".
    #
    #   Step 2: For each group, accumulate two counts: every parameter's
    #           numel() (total) and, only for parameters with
    #           requires_grad == True, its numel() again (trainable).
    #
    #   Step 3: Also accumulate overall "total" and "trainable" across
    #           both groups.
    pass


def print_parameter_summary(summary):
    """Pretty-print the dict returned by parameter_summary()."""
    print(f"{'Group':<12} {'Total':>14} {'Trainable':>14}")
    print("-" * 42)
    print(f"{'Encoder':<12} {summary['encoder_total']:>14,} {summary['encoder_trainable']:>14,}")
    print(f"{'Head':<12} {summary['head_total']:>14,} {summary['head_trainable']:>14,}")
    print("-" * 42)
    print(f"{'Total':<12} {summary['total']:>14,} {summary['trainable']:>14,}")
    pct = 100 * summary["trainable"] / summary["total"]
    print(f"\n{pct:.2f}% of parameters are trainable.")


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true",
                         help="Apply freeze_encoder() before reporting parameter counts.")
    args = parser.parse_args()

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_classes = 3

    model = load_pretrained_segformer(model_name, num_classes)

    if args.freeze:
        model = freeze_encoder(model)
        print(f"Loaded {model_name}, encoder frozen.\n")
    else:
        print(f"Loaded {model_name}, nothing frozen (default: everything trainable).\n")

    summary = parameter_summary(model)
    print_parameter_summary(summary)

    dummy_input = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
    print(f"\nLogits shape: {list(outputs.logits.shape)}  <- note: NOT (B, num_classes, 512, 512)")
    print("SegFormer's decode head outputs at 1/4 resolution -- upsample before")
    print("computing predictions or metrics (see evaluate.py).")

    print("\n--- Model verified. ---")
