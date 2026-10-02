"""
Exercise 08 - Distillation Loss
=============================================
Implements the classic teacher-student distillation loss (Hinton,
Vinyals & Dean, 2015): a temperature-softened KL term plus ordinary
cross-entropy against the true labels. No training loop, no real
teacher/student model -- just the loss math, checked against toy logits.

Run this script to verify:
    python src/distillation_loss.py
"""

import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    Combine a temperature-softened distillation term with ordinary
    cross-entropy against the true labels. See README Section 3 for the
    full explanation.

    Args:
        student_logits (Tensor): shape (batch, num_classes), raw logits
            from the small model.
        teacher_logits (Tensor): shape (batch, num_classes), raw logits
            from the large model, same shape as student_logits.
        labels (Tensor): shape (batch,), integer class indices.
        temperature (float): softens both distributions before the KL
            term -- higher values reveal more of the teacher's
            "dark knowledge" about class similarity.
        alpha (float): weight on the soft (distillation) loss; the hard
            (cross-entropy) loss gets weight (1 - alpha).

    Returns:
        Tensor: scalar combined loss.
    """
    # TODO: Implement the two-term distillation loss.
    #
    #   Step 1 (soft loss -- the distillation term): the KL divergence
    #           between the student's and teacher's temperature-softened
    #           distributions. Use F.log_softmax(student_logits /
    #           temperature, dim=-1) and F.softmax(teacher_logits /
    #           temperature, dim=-1) as the two arguments to
    #           F.kl_div(..., reduction="batchmean"), then multiply the
    #           result by temperature ** 2 (this rescaling keeps
    #           gradients comparable across different T -- see Hinton et
    #           al., 2015).
    #
    #   Step 2 (hard loss -- the ordinary term): plain
    #           F.cross_entropy(student_logits, labels), using the
    #           student's raw, un-softened logits.
    #
    #   Step 3: return alpha times the soft loss plus (1 - alpha) times
    #           the hard loss -- alpha is the soft-loss weight.
    pass


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import torch

    # A toy 4-example, 3-class batch -- same 3 classes as PhenoBench
    # (background / crop / weed), standing in for one "big" and one
    # "small" model's output on the same inputs. No real model runs
    # here -- see the README for why that's intentional.
    student_logits = torch.tensor([
        [2.0, 0.5, 0.1],
        [0.3, 1.5, 0.4],
        [0.2, 0.3, 1.2],
        [1.0, 0.9, 0.2],
    ])
    teacher_logits = torch.tensor([
        [5.0, 1.0, 0.0],
        [0.5, 4.0, 0.2],
        [0.1, 0.2, 3.0],
        [2.0, 1.8, 0.1],
    ])
    labels = torch.tensor([0, 1, 2, 0])

    loss = distillation_loss(student_logits, teacher_logits, labels)
    print(f"distillation_loss (alpha=0.5, T=4.0): {loss.item():.4f}")

    expected = 0.5676
    assert abs(loss.item() - expected) < 1e-3, (
        f"Expected ~{expected}, got {loss.item():.4f} -- check the KL term, "
        f"the temperature ** 2 scaling, or the alpha weighting."
    )
    print("Matches the known reference value.")

    # Sanity checks: the two extremes should reduce to a single term.
    hard_only = distillation_loss(student_logits, teacher_logits, labels, alpha=0.0)
    hard_reference = F.cross_entropy(student_logits, labels)
    assert torch.isclose(hard_only, hard_reference, atol=1e-5), (
        "alpha=0.0 should reduce to plain cross-entropy."
    )

    soft_only = distillation_loss(student_logits, teacher_logits, labels, alpha=1.0)
    soft_reference = F.kl_div(
        F.log_softmax(student_logits / 4.0, dim=-1),
        F.softmax(teacher_logits / 4.0, dim=-1),
        reduction="batchmean",
    ) * 4.0 ** 2
    assert torch.isclose(soft_only, soft_reference, atol=1e-5), (
        "alpha=1.0 should reduce to the temperature-scaled KL term alone."
    )
    print("alpha=0.0 and alpha=1.0 edge cases check out.")

    print("\n--- Loss verified. ---")
