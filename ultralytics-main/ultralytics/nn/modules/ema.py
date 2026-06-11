# Ultralytics AGPL-3.0 License

"""
Efficient Multi-scale Attention (EMA) module.
From: "Efficient Multi-Scale Attention Module with Cross-Spatial Learning", ICASSP 2023.

Placed in the NECK (not backbone) to avoid disrupting pretrained features.
Uses parallel group convolutions at multiple kernel sizes to capture
fracture patterns at different scales.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention.

    Args:
        c1 (int): Input channels.
        c2 (int): Unused (Ultralytics YAML compat).
        groups (int): Number of channel groups for multi-scale decomposition.
    """

    def __init__(self, c1, c2=None, groups=4):
        super().__init__()
        self.groups = groups
        assert c1 % groups == 0, f"Channels {c1} must be divisible by groups {groups}"
        gc = c1 // groups

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.softmax = nn.Softmax(dim=-1)
        self.gn = nn.GroupNorm(groups, c1)

        # Multi-scale 1D convolutions for cross-spatial attention
        self.conv1x1 = nn.Conv2d(c1, c1, 1)
        self.conv3x3 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1)
        self.conv5x5 = nn.Conv2d(c1, c1, 5, 1, 2, groups=c1)

        self.conv_squeeze = nn.Conv2d(3, 3, 7, 1, 3)
        self.conv_out = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Group normalization for multi-scale decomposition
        x_gn = self.gn(x)

        # Multi-scale feature extraction
        x1 = self.conv1x1(x_gn)
        x3 = self.conv3x3(x_gn)
        x5 = self.conv5x5(x_gn)

        # Cross-spatial attention weights
        x_cat = torch.stack([x1, x3, x5], dim=1)  # (b, 3, c, h, w)
        x_avg = x_cat.mean(dim=2)  # (b, 3, h, w)
        attn = self.conv_squeeze(x_avg)  # (b, 3, h, w)
        attn = self.softmax(attn)  # spatial attention weights for each scale

        # Weighted combination
        out = (x1 * attn[:, 0:1] + x3 * attn[:, 1:2] + x5 * attn[:, 2:3])
        out = self.conv_out(out)

        return x + out  # residual connection