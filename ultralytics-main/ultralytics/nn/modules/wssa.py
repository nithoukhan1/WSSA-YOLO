# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Wavelet Sub-band Selective Attention (WSSA) module for fracture detection."""

import torch
import torch.nn as nn


class _SubBandSE(nn.Module):
    """Squeeze-and-Excitation for one wavelet sub-band."""

    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class WSSA(nn.Module):
    """Wavelet Sub-band Selective Attention.

    Decomposes features via 2D Haar DWT into four frequency sub-bands (LL, LH, HL, HH),
    applies independent SE channel attention to each, reconstructs via inverse DWT,
    and adds via a learnable residual connection.

    Args:
        c1 (int): Input channels.
        c2 (int): Unused (Ultralytics YAML compat — parser always passes 2 channel args).
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        self.c = c1
        self.se_ll = _SubBandSE(c1)
        self.se_lh = _SubBandSE(c1)
        self.se_hl = _SubBandSE(c1)
        self.se_hh = _SubBandSE(c1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _dwt(x):
        """2D Haar DWT."""
        a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
        c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
        return (a+b+c+d)*0.5, (a-b+c-d)*0.5, (a+b-c-d)*0.5, (a-b-c+d)*0.5

    @staticmethod
    def _idwt(ll, lh, hl, hh):
        """Inverse 2D Haar DWT."""
        B, C, H2, W2 = ll.shape
        x = ll.new_zeros(B, C, H2*2, W2*2)
        x[:, :, 0::2, 0::2] = (ll+lh+hl+hh)*0.5
        x[:, :, 0::2, 1::2] = (ll-lh+hl-hh)*0.5
        x[:, :, 1::2, 0::2] = (ll+lh-hl-hh)*0.5
        x[:, :, 1::2, 1::2] = (ll-lh-hl+hh)*0.5
        return x

    def forward(self, x):
        _, _, H, W = x.shape
        ph, pw = H % 2, W % 2
        xp = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        ll, lh, hl, hh = self._dwt(xp)
        rec = self._idwt(self.se_ll(ll), self.se_lh(lh), self.se_hl(hl), self.se_hh(hh))
        if ph or pw:
            rec = rec[:, :, :H, :W]
        return x + self.alpha * rec