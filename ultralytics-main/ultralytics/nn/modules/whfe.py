"""Wavelet-domain High-Frequency Enhancement (WHFE) module.

Decomposes feature maps via 2D Haar DWT into 4 frequency bands (LL, LH, HL, HH),
applies per-channel learnable gating to the 3 high-frequency bands, and
reconstructs via inverse DWT. The DWT/IDWT transforms are parameter-free.
Only the gates (3 * C scalars per insertion point) are learnable.

Reference design philosophy:
    - Fracture lines are high-frequency edge structures
    - Parameter-light learning so rare-class signal is sufficient
    - Identity at init (gates = 1.0) so pretrained weights flow through cleanly
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarTransform(nn.Module):
    """2D Haar DWT and IDWT via grouped convolution.
    
    Uses orthonormal Haar filters with 1/sqrt(2) normalization for
    perfect reconstruction when gates are identity.
    """
    def __init__(self):
        super().__init__()
        h = 1.0 / math.sqrt(2.0)
        # Four 2x2 Haar filters: LL, LH (vertical), HL (horizontal), HH (diagonal)
        filters = torch.tensor([
            [[h,  h], [ h,  h]],   # LL — low-low
            [[h,  h], [-h, -h]],   # LH — captures vertical edges
            [[h, -h], [ h, -h]],   # HL — captures horizontal edges
            [[h, -h], [-h,  h]],   # HH — captures diagonal edges
        ])  # shape: [4, 2, 2]
        # Register as [4, 1, 2, 2] for conv2d with per-channel grouping
        self.register_buffer('filters', filters.unsqueeze(1))

    def dwt(self, x):
        """Forward DWT: [B,C,H,W] -> [B,C,4,H/2,W/2]"""
        B, C, H, W = x.shape
        # Reshape so each channel is processed independently
        x_flat = x.reshape(B * C, 1, H, W)
        y = F.conv2d(x_flat, self.filters, stride=2)  # [B*C, 4, H/2, W/2]
        return y.reshape(B, C, 4, H // 2, W // 2)

    def idwt(self, y):
        """Inverse DWT: [B,C,4,H,W] -> [B,C,2H,2W]"""
        B, C, _, H, W = y.shape
        y_flat = y.reshape(B * C, 4, H, W)
        x = F.conv_transpose2d(y_flat, self.filters, stride=2)  # [B*C, 1, 2H, 2W]
        return x.reshape(B, C, 2 * H, 2 * W)


class WHFE(nn.Module):
    """Wavelet-domain High-Frequency Enhancement.
    
    Args:
        c1: input/output channels
        c2: unused (kept for YAML signature compatibility)
    
    Parameter count: 3 * c1 (channel-wise gates for LH, HL, HH bands)
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        self.c = c1
        self.haar = HaarTransform()
        # Per-channel gates for 3 high-frequency bands
        # Init to 1.0 → at init WHFE acts as exact identity
        self.gate_lh = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.gate_hl = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.gate_hh = nn.Parameter(torch.ones(1, c1, 1, 1))
        # Track whether module is enabled (env-var gated)
        self._enabled = os.environ.get('USE_WHFE', '0') == '1'

    def forward(self, x):
        # If module disabled by env var, act as pure identity (no DWT cost)
        if not self._enabled:
            return x
        # DWT decomposition
        y = self.haar.dwt(x)  # [B, C, 4, H/2, W/2]
        ll = y[:, :, 0]                            # low-low (passes through)
        lh = y[:, :, 1] * self.gate_lh             # vertical detail (gated)
        hl = y[:, :, 2] * self.gate_hl             # horizontal detail (gated)
        hh = y[:, :, 3] * self.gate_hh             # diagonal detail (gated)
        # Stack and inverse transform
        y = torch.stack([ll, lh, hl, hh], dim=2)   # [B, C, 4, H/2, W/2]
        return self.haar.idwt(y)