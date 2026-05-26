# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Frequency-Aware Feature Fusion Upsampling (FreqFusionUp).

Adapted from: Chen et al., "Frequency-aware Feature Fusion for Dense Image
Prediction", IEEE TPAMI 2024. (https://github.com/Linwei-Chen/FreqFusion)

Replaces nn.Upsample + Concat in the YOLO neck with a frequency-aware
alternative. Takes two inputs (low-res deep features + high-res skip features),
applies Adaptive Low-Pass Filtering to the deep features before upsampling,
enhances the skip features with Adaptive High-Pass Filtering to recover
boundary details, and concatenates both.

This complements WSSA (backbone) which extracts frequency-selective features —
FreqFusionUp (neck) preserves those features during multi-scale fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqFusionUp(nn.Module):
    """Frequency-Aware Feature Fusion Upsampling.

    Replaces nn.Upsample + Concat in the YOLO PAN-FPN neck.
    Accepts a list of two tensors [lr_feat, hr_feat] (same format as Concat).
    Output channels = lr_channels + hr_channels (same as Upsample + Concat).

    Three components from the FreqFusion paper (TPAMI 2024):
      1. ALPF — Adaptive Low-Pass Filter: smooths lr_feat before upsampling
         to reduce intra-class inconsistency (noisy high-freq artifacts).
      2. Offset-guided upsample: learns sampling offsets for the upsampled
         grid so the upsampling adapts to local feature structure.
      3. AHPF — Adaptive High-Pass Filter: enhances hr_feat boundaries
         by extracting and gating high-frequency residuals.

    Args:
        c_lr (int): Channels of the low-resolution (deeper) input.
        c_hr (int): Channels of the high-resolution (skip) input.
    """

    def __init__(self, c_lr, c_hr):
        super().__init__()

        # ── ALPF: Adaptive Low-Pass Filter for lr_feat ──────────────
        # Channel gating: learns which channels carry useful low-freq info
        # vs noisy high-freq that should be suppressed before upsampling.
        self.alpf_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_lr, max(c_lr // 4, 8), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c_lr // 4, 8), c_lr, 1, bias=False),
            nn.Sigmoid(),
        )
        # Depthwise 5x5 conv acts as a learned spatially-adaptive smoother.
        # Larger kernel = stronger low-pass effect for fracture region consistency.
        self.alpf_smooth = nn.Sequential(
            nn.Conv2d(c_lr, c_lr, 5, 1, 2, groups=c_lr, bias=False),
            nn.BatchNorm2d(c_lr),
        )

        # ── Offset Generator: learnable upsampling offsets ──────────
        # Predicts 2D offsets for each position in the 2x upsampled grid.
        # Unlike fixed bilinear interpolation, these offsets adapt to local
        # feature structure — fracture edges get sharper sampling.
        scale = 2
        self.scale = scale
        self.offset = nn.Conv2d(c_lr, 2 * scale * scale, 3, 1, 1, bias=False)
        nn.init.trunc_normal_(self.offset.weight, std=0.02)

        # ── AHPF: Adaptive High-Pass Filter for hr_feat ────────────
        # Extracts high-frequency boundary details from the skip connection
        # that were lost during downsampling. A learned spatial gate controls
        # how much enhancement to apply at each position.
        self.ahpf_gate = nn.Sequential(
            nn.Conv2d(c_hr, c_hr, 3, 1, 1, groups=c_hr, bias=False),
            nn.BatchNorm2d(c_hr),
            nn.Sigmoid(),
        )
        # Learnable enhancement strength (starts moderate at 0.3)
        self.beta = nn.Parameter(torch.tensor(0.3))

    def _offset_upsample(self, x, offsets):
        """Upsample x by self.scale using base grid + learned offsets."""
        B, C, H, W = x.shape
        s = self.scale
        oH, oW = H * s, W * s

        # Create base grid in [-1, 1] (normalized coords for grid_sample)
        grid_y = torch.linspace(-1, 1, oH, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(-1, 1, oW, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Reshape offsets: (B, 2*s*s, H, W) → (B, oH, oW, 2)
        offsets = offsets.view(B, 2, s, s, H, W)
        offsets = offsets.permute(0, 1, 4, 2, 5, 3).reshape(B, 2, oH, oW)
        offsets = offsets.permute(0, 2, 3, 1)

        # Bound offsets to prevent sampling too far from base position
        offsets = offsets.tanh() * (2.0 / max(oH, oW))

        grid = base_grid + offsets
        return F.grid_sample(
            x, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )

    def forward(self, x):
        """Forward pass. x is a list [lr_feat, hr_feat] (like Concat).

        lr_feat: (B, c_lr, H, W) — low-res features from deeper layer
        hr_feat: (B, c_hr, 2H, 2W) — high-res features from skip connection

        Returns: (B, c_lr + c_hr, 2H, 2W) — frequency-enhanced concatenation
        """
        lr_feat, hr_feat = x[0], x[1]

        # ── 1. ALPF: smooth lr_feat before upsampling ──
        gate = self.alpf_gate(lr_feat)
        lr_gated = lr_feat * gate
        lr_smooth = self.alpf_smooth(lr_gated) + lr_gated  # residual smoothing

        # ── 2. Offset-guided upsampling ──
        offsets = self.offset(lr_smooth)
        lr_up = self._offset_upsample(lr_smooth, offsets)

        # ── 3. AHPF: enhance hr_feat boundaries ──
        hr_blur = F.avg_pool2d(
            F.pad(hr_feat, [1, 1, 1, 1], mode="reflect"), 3, 1
        )
        hr_high = hr_feat - hr_blur
        ahpf_g = self.ahpf_gate(hr_feat)
        hr_enhanced = hr_feat + self.beta * ahpf_g * hr_high

        # ── 4. Concatenate ──
        return torch.cat([lr_up, hr_enhanced], dim=1)