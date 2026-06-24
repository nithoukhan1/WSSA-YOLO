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

    Idempotent and resume-safe. Attaches strictly to the parent DetectionModel
    wrapper to avoid polluting the sequential layer stack and to prevent 
    forward-pass argument corruption in the C2PSA block.
    """
    # 1. Isolate the parent DetectionModel
    # (If wrapped in high-level YOLO class, unwrap it)
    detection_model = model.model if type(model).__name__ == 'YOLO' else model

    # 2. Locate the Sequential layer stack just to get the C2PSA layer
    if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
        layers = detection_model.model
    else:
        raise RuntimeError(f"Could not locate Sequential stack in {type(detection_model).__name__}")

    c2psa_layer = layers[C2PSA_LAYER_IDX]

    # 3. KEY FIX: Attach head to the PARENT DetectionModel, not Sequential, not C2PSA.
    if hasattr(detection_model, '_mcaux_head') and isinstance(detection_model._mcaux_head, MCAuxHead):
        head = detection_model._mcaux_head
    else:
        head = MCAuxHead(in_channels=C2PSA_OUT_CHANNELS, num_classes=num_classes)
        head = head.to(next(detection_model.parameters()).device)
        detection_model.add_module('_mcaux_head', head)

    # 4. Forward hook stores logits directly on the DetectionModel
    def hook(module, inputs, output):
        if mcaux_enabled() and module.training:
            detection_model._mcaux_last_logits = detection_model._mcaux_head(output)
        else:
            detection_model._mcaux_last_logits = None

    # 5. Safe hook registration
    if hasattr(c2psa_layer, '_mcaux_hook_handle'):
        try:
            c2psa_layer._mcaux_hook_handle.remove()
        except Exception:
            pass

    handle = c2psa_layer.register_forward_hook(hook)
    c2psa_layer._mcaux_hook_handle = handle

    return head, handle


def get_last_mcaux_logits(model: nn.Module):
    """Retrieve logits stored on the DetectionModel by the forward hook."""
    detection_model = model.model if type(model).__name__ == 'YOLO' else model
    return getattr(detection_model, '_mcaux_last_logits', None)