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
    """Attach MCAux head and patch the C2PSA layer's forward method.

    Idempotent and resume-safe. Uses direct forward-method patching instead of
    PyTorch forward hooks (hooks caused dict-input issues in Ultralytics 8.4.50's
    _predict_once iteration). The head is stored on the C2PSA layer itself so
    Ultralytics' Sequential iteration never sees it.

    Args:
        model: a YOLO DetectionModel wrapper (has .model.model Sequential)
        num_classes: number of classes from the dataset
    Returns:
        the MCAuxHead module (hook_handle is None — we use patching, not hooks)
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

    # Create or reuse the MCAux head. Stored as a submodule of c2psa_layer
    # so its parameters are picked up by model.parameters() (optimizer)
    # and saved with checkpoints. _predict_once does NOT recurse into
    # c2psa_layer's children, so the head won't be iterated over.
    if hasattr(c2psa_layer, '_mcaux_head') and isinstance(c2psa_layer._mcaux_head, MCAuxHead):
        head = c2psa_layer._mcaux_head
    else:
        head = MCAuxHead(in_channels=C2PSA_OUT_CHANNELS, num_classes=num_classes)
        head = head.to(next(inner.parameters()).device)
        c2psa_layer.add_module('_mcaux_head', head)

    # Monkey-patch forward: save the original method, replace with a wrapper
    # that calls original then runs MCAux head on the output.
    # Idempotent: if already patched, the cached _mcaux_original_forward exists
    # and we reuse it instead of double-wrapping.
    if not hasattr(c2psa_layer, '_mcaux_original_forward'):
        c2psa_layer._mcaux_original_forward = c2psa_layer.forward

    original_forward = c2psa_layer._mcaux_original_forward

    def patched_forward(x):
        result = original_forward(x)
        # Only compute and store logits during training (not eval/inference)
        if mcaux_enabled() and c2psa_layer.training:
            c2psa_layer._mcaux_last_logits = c2psa_layer._mcaux_head(result)
        else:
            c2psa_layer._mcaux_last_logits = None
        return result

    # Install the patched forward on the instance (instance attribute overrides
    # class-level forward for this specific c2psa_layer)
    c2psa_layer.forward = patched_forward

    # Store a reference on inner so the loss can locate the c2psa_layer
    inner._mcaux_c2psa_layer = c2psa_layer

    # Return head and a None placeholder (no hook handle since we don't use hooks)
    return head, None


def get_last_mcaux_logits(model: nn.Module):
    """Retrieve logits stored on the C2PSA layer by patched forward.

    Returns None if MCAux is disabled, if the model is in eval mode,
    or if forward hasn't run yet.
    """
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        inner = model.model
    else:
        inner = model
    c2psa_layer = getattr(inner, '_mcaux_c2psa_layer', None)
    if c2psa_layer is None:
        return None
    return getattr(c2psa_layer, '_mcaux_last_logits', None)