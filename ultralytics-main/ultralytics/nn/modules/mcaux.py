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

    Idempotent and resume-safe:
    - If model already has _mcaux_head submodule (loaded from checkpoint),
      reuse it. Otherwise create a new one.
    - Always register a fresh forward hook (hooks are not pickled with the
      model, so they must be re-registered on every load/resume).

    Args:
        model: a YOLO DetectionModel (top-level wrapper, has .model attribute
               which is the nn.Sequential of layers)
        num_classes: number of classes from the dataset
    Returns:
        tuple(head, hook_handle)
    """
    # The top-level YOLO model wraps a DetectionModel. Get the inner one.
    # Ultralytics: model.model is the DetectionModel; model.model.model is the Sequential.
    # We need the Sequential of layers to index into.
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
    else:
        inner = model

    # inner.model is the Sequential
    if hasattr(inner, 'model') and isinstance(inner.model, nn.Sequential):
        layers = inner.model
    elif isinstance(inner, nn.Sequential):
        layers = inner
    else:
        raise RuntimeError(
            f'Could not locate Sequential of layers in model. '
            f'Got type {type(inner).__name__}'
        )

    c2psa_layer = layers[C2PSA_LAYER_IDX]

    # Re-use head if it exists (resume case), else create new
    if hasattr(inner, '_mcaux_head') and isinstance(inner._mcaux_head, MCAuxHead):
        head = inner._mcaux_head
    else:
        head = MCAuxHead(in_channels=C2PSA_OUT_CHANNELS, num_classes=num_classes)
        head = head.to(next(inner.parameters()).device)
        inner.add_module('_mcaux_head', head)

    # Always (re-)register the forward hook. Hooks are not saved by pickle.
    def hook(module, inputs, output):
        # output: C2PSA feature map [B, 512, H, W]
        if mcaux_enabled() and module.training:
            # Store logits on the inner model so the loss can retrieve them.
            inner._mcaux_last_logits = inner._mcaux_head(output)
        else:
            inner._mcaux_last_logits = None

    # Remove any previously-registered MCAux hook before adding a new one
    # (matters for resume — avoid duplicate hooks).
    if hasattr(c2psa_layer, '_mcaux_hook_handle'):
        try:
            c2psa_layer._mcaux_hook_handle.remove()
        except Exception:
            pass

    handle = c2psa_layer.register_forward_hook(hook)
    c2psa_layer._mcaux_hook_handle = handle  # store so we can remove on re-attach

    return head, handle


def get_last_mcaux_logits(model: nn.Module):
    """Retrieve the logits stored on the model by the forward hook.

    Returns None if MCAux is disabled, if the model is in eval mode,
    or if the hook hasn't fired yet.
    """
    # Same model-traversal pattern as attach_mcaux_to_model
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
    else:
        inner = model
    return getattr(inner, '_mcaux_last_logits', None)