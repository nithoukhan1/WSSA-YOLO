"""Effective-number class weighting utilities for GRAZPEDWRI-DX.

This module provides:

1. The original effective-number weights proposed by Cui et al.
2. A detector-specific capped relative weighting strategy for
   foreground Task-Aligned Assigner anchors.

The detector-specific weights must not be passed directly as
BCEWithLogitsLoss(pos_weight=...). They are intended to multiply the
complete classification loss of foreground anchors after target
assignment.

Reference:
    Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019).
    Class-Balanced Loss Based on Effective Number of Samples.
    CVPR 2019.
"""

import os

import torch


# Original pre-augmentation training annotation counts.
#
# The deterministic brightness augmentation duplicates every original
# annotation once. We therefore use the original counts instead of
# treating correlated augmented copies as independent observations.
#
# Class order:
#   0: boneanomaly
#   1: bonelesion
#   2: foreignbody
#   3: fracture
#   4: metal
#   5: periostealreaction
#   6: pronatorsign
#   7: softtissue
#   8: text
CLASS_COUNTS = [
    368,
    52,
    16,
    25224,
    1134,
    4818,
    816,
    632,
    33176,
]


def get_cb_weights(
    beta: float = 0.999,
    num_classes: int = 9,
) -> torch.Tensor:
    """Return normalized effective-number weights.

    The effective number for class ``c`` is:

        E_c = (1 - beta ** n_c) / (1 - beta)

    The corresponding class weight is:

        w_c = (1 - beta) / (1 - beta ** n_c)

    The returned weights are normalized so their sum equals the number
    of classes. This function is retained for analysis and comparison.

    These values must not be passed directly as BCE ``pos_weight`` in
    the corrected detector implementation.

    Args:
        beta: Effective-number smoothing coefficient in ``[0, 1)``.
        num_classes: Expected number of dataset classes.

    Returns:
        A float32 tensor with shape ``[num_classes]``.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(
            f"beta must be in [0, 1), received {beta}"
        )

    if len(CLASS_COUNTS) != num_classes:
        raise ValueError(
            "CLASS_COUNTS contains "
            f"{len(CLASS_COUNTS)} entries, "
            f"but num_classes={num_classes}."
        )

    counts = torch.tensor(
        CLASS_COUNTS,
        dtype=torch.float32,
    )

    beta_tensor = torch.tensor(
        beta,
        dtype=torch.float32,
    )

    effective_denominator = (
        1.0 - torch.pow(beta_tensor, counts)
    )

    weights = (
        (1.0 - beta)
        / effective_denominator
    )

    weights = (
        weights
        / weights.sum()
        * num_classes
    )

    return weights


def get_cb_foreground_weights(
    beta: float = 0.999,
    num_classes: int = 9,
    min_weight: float = 1.0,
    max_weight: float = 2.0,
) -> torch.Tensor:
    """Return capped relative weights for foreground detector anchors.

    The smallest effective-number weight is treated as the reference
    value of 1.0. Minority-class weights are increased relative to this
    reference and capped to prevent extreme gradients.

    Proposed FWNet detector weighting:

        relative_c = raw_c / min(raw)
        final_c = clamp(relative_c, min_weight, max_weight)

    With the current GRAZPEDWRI-DX counts and default settings, fracture
    and text remain at 1.0, while minority classes receive weights up to
    2.0.

    Background anchors must remain at weight 1.0. That behavior will be
    implemented separately in the detection classification-loss code.

    Args:
        beta: Effective-number smoothing coefficient.
        num_classes: Expected number of dataset classes.
        min_weight: Minimum allowed foreground weight.
        max_weight: Maximum allowed foreground weight.

    Returns:
        A float32 tensor with shape ``[num_classes]``.
    """
    if min_weight <= 0.0:
        raise ValueError(
            "min_weight must be greater than zero."
        )

    if max_weight < min_weight:
        raise ValueError(
            "max_weight must be greater than or equal "
            "to min_weight."
        )

    effective_weights = get_cb_weights(
        beta=beta,
        num_classes=num_classes,
    )

    relative_weights = (
        effective_weights
        / effective_weights.min()
    )

    foreground_weights = relative_weights.clamp(
        min=min_weight,
        max=max_weight,
    )

    return foreground_weights


def cb_loss_enabled() -> bool:
    """Return True when USE_CBLOSS is enabled."""
    return os.environ.get(
        "USE_CBLOSS",
        "0",
    ) == "1"