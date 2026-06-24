"""Multi-task Classification Auxiliary head (MCAux) for FWNet-YOLO.

Attaches an image-level multi-label classification head to the model's backbone
output (after the C2PSA block at layer 10). During training, predicts which
classes are present anywhere in the image. The shared backbone gets dense
supervision (one signal per image per class) that compensates for sparse
rare-class box supervision.

Design:
    - The aux head is stored on the model as `model._mcaux_head` so it saves
      and resumes via the standard YOLO checkpointing.
    - A forward hook on the C2PSA layer (index 10) intercepts the feature map
      during each training step and stores logits on `model._mcaux_last_logits`.
    - The loss class reads `model._mcaux_last_logits` and folds the auxiliary
      BCE loss into the total. λ=0.5 keeps it auxiliary, not dominant.
    - At inference (`model.eval()`), the hook's `module.training` check is
      False, so logits aren't stored and aux loss isn't computed.

Env-var gated by USE_MCAUX. Default OFF.
"""

import os
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (fixed for this paper — don't change without re-ablating)
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA = 0.5            # auxiliary loss weight in total loss
HIDDEN_DIM = 256        # MLP hidden dimension
DROPOUT_P = 0.2         # dropout in MLP
C2PSA_OUT_CHANNELS = 512  # at YOLOv11s scale (width_mult=0.5 applied to 1024)
C2PSA_LAYER_IDX = 10    # index of C2PSA layer in YOLOv11 architecture


def mcaux_enabled() -> bool:
    """Returns True if USE_MCAUX=1 in env vars."""
    return os.environ.get('USE_MCAUX', '0') == '1'


def get_lambda() -> float:
    """Auxiliary loss weight."""
    return LAMBDA


# ─────────────────────────────────────────────────────────────────────────────
# The auxiliary head module
# ─────────────────────────────────────────────────────────────────────────────
class MCAuxHead(nn.Module):
    """Image-level multi-label classification head.
    
    Args:
        in_channels: input feature channels (512 for YOLOv11s after width_mult)
        num_classes: number of classes (9 for GRAZPEDWRI-DX)
    """
    def __init__(self, in_channels: int = C2PSA_OUT_CHANNELS, num_classes: int = 9):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, HIDDEN_DIM)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
        # Init final layer near zero so aux loss starts small and doesn't dominate
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.gap(feat).flatten(1)         # [B, C]
        x = self.act(self.fc1(x))             # [B, hidden]
        x = self.dropout(x)
        return self.fc2(x)                    # [B, num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Attachment: idempotent function called by the loss class
# ─────────────────────────────────────────────────────────────────────────────
def attach_mcaux_to_model(model: nn.Module, num_classes: int = 9):
    """Attach MCAux head and forward hook to the model.

    Idempotent and resume-safe. The MCAuxHead is stored on the C2PSA LAYER
    (not on the DetectionModel) so it doesn't appear in Ultralytics'
    _predict_once iteration (which requires every child to have a .f attribute).

    Args:
        model: a YOLO DetectionModel wrapper (has .model.model Sequential)
        num_classes: number of classes from the dataset
    Returns:
        tuple(head, hook_handle)
    """
    # Traverse to the inner Sequential of layers
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
    else:
        inner = model

    if hasattr(inner, 'model') and isinstance(inner.model, nn.Sequential):
        layers = inner.model
    elif isinstance(inner, nn.Sequential):
        layers = inner
    else:
        raise RuntimeError(
            f'Could not locate Sequential of layers. Got type {type(inner).__name__}'
        )

    c2psa_layer = layers[C2PSA_LAYER_IDX]

    # ── KEY FIX: store head on c2psa_layer, NOT on the DetectionModel ──
    # Storing on the DetectionModel via add_module causes _predict_once to
    # iterate over MCAuxHead and crash looking for attribute .f.
    # Storing on c2psa_layer keeps it as a submodule of that specific layer,
    # so it's saved with the checkpoint but never seen by _predict_once.
    if hasattr(c2psa_layer, '_mcaux_head') and isinstance(c2psa_layer._mcaux_head, MCAuxHead):
        # Resume case: head already exists, reuse it
        head = c2psa_layer._mcaux_head
    else:
        # Fresh training: create and attach to c2psa_layer
        head = MCAuxHead(in_channels=C2PSA_OUT_CHANNELS, num_classes=num_classes)
        head = head.to(next(inner.parameters()).device)
        c2psa_layer.add_module('_mcaux_head', head)

    # Hook stores logits on c2psa_layer (transient, not pickled)
    def hook(module, inputs, output):
        if mcaux_enabled() and module.training:
            module._mcaux_last_logits = module._mcaux_head(output)
        else:
            module._mcaux_last_logits = None

    # Remove old hook before re-registering (handles resume case)
    if hasattr(c2psa_layer, '_mcaux_hook_handle'):
        try:
            c2psa_layer._mcaux_hook_handle.remove()
        except Exception:
            pass

    handle = c2psa_layer.register_forward_hook(hook)
    c2psa_layer._mcaux_hook_handle = handle

    # Store c2psa_layer reference on inner model so loss() can find it
    inner._mcaux_c2psa_layer = c2psa_layer

    return head, handle


def get_last_mcaux_logits(model: nn.Module):
    """Retrieve logits stored on the C2PSA layer by the forward hook."""
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
    else:
        inner = model
    c2psa_layer = getattr(inner, '_mcaux_c2psa_layer', None)
    if c2psa_layer is None:
        return None
    return getattr(c2psa_layer, '_mcaux_last_logits', None)