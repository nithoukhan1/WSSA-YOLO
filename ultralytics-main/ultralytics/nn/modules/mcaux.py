"""Multi-task Classification Auxiliary head (MCAux) for FWNet-YOLO.

Attaches a small image-level classification head to the model's backbone output
(after the C2PSA block). During training, predicts per-class image-level
presence labels (BCE multi-label). During inference, this head is bypassed.

Purpose: provide abundant image-level supervision (~14,204 images × 9 classes
= 127,836 signals) to compensate for sparse rare-class box supervision
(bonelesion has only 52 box instances).

The auxiliary head is attached via forward hook on the C2PSA layer (index 10)
rather than modifying the architecture. This keeps the model topologically
identical to baseline YOLOv11s — no INDEX_MAP needed, no weight transfer issues.
"""

import os
import torch
import torch.nn as nn


class MCAuxHead(nn.Module):
    """Image-level classification head: backbone features → per-class logits.
    
    Args:
        in_channels: number of channels from C2PSA output (512 for YOLOv11s)
        num_classes: number of classes (9 for GRAZPEDWRI-DX)
        hidden_dim: MLP hidden dimension (default 256)
    """
    def __init__(self, in_channels: int = 512, num_classes: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize so logits start near zero — auxiliary loss won't dominate at init
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: backbone feature map, shape [B, C, H, W]
        Returns:
            logits: shape [B, num_classes]
        """
        x = self.gap(feat)              # [B, C, 1, 1]
        x = x.flatten(1)                # [B, C]
        x = self.act(self.fc1(x))       # [B, hidden_dim]
        x = self.dropout(x)
        x = self.fc2(x)                 # [B, num_classes]
        return x


def mcaux_enabled() -> bool:
    """Returns True if USE_MCAUX=1 in env vars."""
    return os.environ.get('USE_MCAUX', '0') == '1'


# Globals for hook-based integration (one model per process during training)
_MCAUX_HEAD = None
_MCAUX_LAST_LOGITS = None
_MCAUX_LAMBDA = 0.5  # auxiliary loss weight


def attach_mcaux_to_model(model: nn.Module, num_classes: int = 9):
    """Attach MCAux head and forward hook to the model's C2PSA layer (index 10).
    
    This is called once after the model is built. The hook captures the C2PSA
    output during forward pass and runs it through MCAuxHead, stashing the
    logits in a module-level global for the loss computation to retrieve.
    
    Args:
        model: the YOLO DetectionModel (model.model is the nn.Sequential)
        num_classes: nc from dataset
    Returns:
        the MCAuxHead module (also stored in _MCAUX_HEAD)
    """
    global _MCAUX_HEAD, _MCAUX_LAST_LOGITS
    
    # YOLOv11s C2PSA output is 512 channels at the 's' scale
    c2psa_layer = model.model[10]
    in_ch = 512  # at YOLOv11s scale; if changing scale, adjust
    
    head = MCAuxHead(in_channels=in_ch, num_classes=num_classes)
    head = head.to(next(model.parameters()).device)
    _MCAUX_HEAD = head
    
    # Register the head as a submodule so it's saved with checkpoints
    # and its parameters appear in the optimizer
    if not hasattr(model, '_mcaux_head'):
        model.add_module('_mcaux_head', head)
    
    def hook(module, inputs, output):
        global _MCAUX_LAST_LOGITS
        if mcaux_enabled() and module.training:
            # output is the C2PSA feature map [B, 512, H, W]
            _MCAUX_LAST_LOGITS = _MCAUX_HEAD(output)
        else:
            _MCAUX_LAST_LOGITS = None
    
    handle = c2psa_layer.register_forward_hook(hook)
    return head, handle


def get_last_mcaux_logits():
    """Retrieve logits from most recent forward pass (or None if disabled/eval)."""
    return _MCAUX_LAST_LOGITS


def get_lambda():
    return _MCAUX_LAMBDA