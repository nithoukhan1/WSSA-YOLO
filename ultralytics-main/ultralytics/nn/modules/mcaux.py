"""Multi-task Classification Auxiliary head (MCAux) for FWNet-YOLO.

Design (clean, no hooks, no monkey-patching):
    - MCAuxHead is created and attached as a submodule of the C2PSA layer
      (layers[10]) BEFORE training starts (called from the training notebook
      before model.train(), so the optimizer picks up the head's parameters).
    - At loss time, the loss class retrieves the head via get_mcaux_head(model)
      and feeds it the P5 feature from preds['feats'][-1].
    - Image-level labels are derived for free from box labels.
    - The auxiliary BCE loss is added to total loss with weight λ=0.5.
    - At inference, the loss class is not called → no aux computation → zero
      inference overhead.

Env-var gated by USE_MCAUX. Default OFF.
"""

import os
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA = 0.5                  # auxiliary loss weight
HIDDEN_DIM = 256              # MLP hidden dim
DROPOUT_P = 0.2
P5_CHANNELS = 512             # YOLOv11s P5 feat channels (after width_mult)
C2PSA_LAYER_IDX = 10          # where we attach as submodule for param visibility


def mcaux_enabled() -> bool:
    return os.environ.get('USE_MCAUX', '0') == '1'


def get_lambda() -> float:
    return LAMBDA


# ─────────────────────────────────────────────────────────────────────────────
# The auxiliary head
# ─────────────────────────────────────────────────────────────────────────────
class MCAuxHead(nn.Module):
    """Image-level multi-label classification head.

    Args:
        in_channels: input feature channels (512 for YOLOv11s P5 feat)
        num_classes: 9 for GRAZPEDWRI-DX
    """
    def __init__(self, in_channels: int = P5_CHANNELS, num_classes: int = 9):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, HIDDEN_DIM)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.gap(feat).flatten(1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model traversal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_layers_sequential(model: nn.Module) -> nn.Sequential:
    """Traverse from any model wrapper (YOLO, DetectionModel) to the
    nn.Sequential of layers. Robust across Ultralytics versions.
    """
    cur = model
    # Walk down via .model attribute until we hit a Sequential
    for _ in range(5):  # depth limit, never goes more than ~3 in practice
        if isinstance(cur, nn.Sequential):
            return cur
        if hasattr(cur, 'model') and isinstance(cur.model, nn.Module):
            cur = cur.model
            continue
        break
    raise RuntimeError(
        f'Could not find Sequential of layers from model type {type(model).__name__}'
    )


def attach_mcaux_to_model(model: nn.Module, num_classes: int = 9) -> MCAuxHead:
    """Create and attach MCAuxHead as a submodule of the C2PSA layer.

    Call this BEFORE model.train() in the training notebook so the head's
    parameters are picked up when the optimizer is set up.

    Idempotent: safe to call multiple times (resume case). Returns the
    existing head if already attached, else creates and attaches.

    Args:
        model: YOLO wrapper, DetectionModel, or Sequential — auto-detected
        num_classes: nc from dataset

    Returns:
        the MCAuxHead instance
    """
    layers = _find_layers_sequential(model)
    c2psa_layer = layers[C2PSA_LAYER_IDX]

    if hasattr(c2psa_layer, '_mcaux_head') and isinstance(c2psa_layer._mcaux_head, MCAuxHead):
        return c2psa_layer._mcaux_head

    head = MCAuxHead(in_channels=P5_CHANNELS, num_classes=num_classes)
    # Move to same device as the model
    head = head.to(next(layers.parameters()).device)
    # Add as submodule of c2psa_layer (NOT of the Sequential, NOT of the DetectionModel).
    # This keeps the head out of _predict_once iteration but still in model.parameters().
    c2psa_layer.add_module('_mcaux_head', head)
    return head


def get_mcaux_head(model: nn.Module):
    """Retrieve the attached MCAux head from any model wrapper.

    Returns None if MCAux is not attached (i.e., USE_MCAUX=0 or attach
    wasn't called).
    """
    try:
        layers = _find_layers_sequential(model)
        c2psa_layer = layers[C2PSA_LAYER_IDX]
        return getattr(c2psa_layer, '_mcaux_head', None)
    except Exception:
        return None