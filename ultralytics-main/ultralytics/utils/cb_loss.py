"""Class-Balanced Effective Number Loss (Cui et al., CVPR 2019).

Computes per-class pos_weight tensors for BCEWithLogitsLoss using the
effective-number-of-samples weighting scheme. Toggled via USE_CBLOSS env var.

Reference:
    Cui, Y., Jia, M., Lin, T-Y., Song, Y., & Belongie, S. (2019).
    Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.
"""

import os
import torch

# Per-class training instance counts for GRAZPEDWRI-DX (Ju et al. augmented split).
# Order matches dataset YAML class index order:
#   0:boneanomaly, 1:bonelesion, 2:foreignbody, 3:fracture, 4:metal,
#   5:periostealreaction, 6:pronatorsign, 7:softtissue, 8:text
CLASS_COUNTS = [368, 52, 16, 25224, 1134, 4818, 816, 632, 33176]


def get_cb_weights(beta: float = 0.999, num_classes: int = 9) -> torch.Tensor:
    """Compute class-balanced weights via effective number of samples.

    Effective number: E_n = (1 - beta^n) / (1 - beta)
    Per-class weight: w_c = (1 - beta) / (1 - beta^n_c)
    Normalized so sum of weights = num_classes.

    Args:
        beta: smoothing in [0, 1). 0.999 is the standard from Cui et al.
        num_classes: sanity check that CLASS_COUNTS length matches model nc.

    Returns:
        Tensor [num_classes] of normalized per-class weights.
    """
    assert len(CLASS_COUNTS) == num_classes, \
        f'CLASS_COUNTS has {len(CLASS_COUNTS)} entries, expected {num_classes}'
    counts = torch.tensor(CLASS_COUNTS, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    return weights


def cb_loss_enabled() -> bool:
    """Returns True if USE_CBLOSS=1 in env vars."""
    return os.environ.get('USE_CBLOSS', '0') == '1'