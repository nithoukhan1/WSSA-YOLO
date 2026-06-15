"""GSConv + VoVGSCSP for Slim-Neck.
Li et al., "Slim-neck by GSConv: a lightweight-design for real-time detector architectures",
Journal of Real-Time Image Processing 21:62 (2024).
"""
import torch
import torch.nn as nn
from .conv import Conv


class GSConv(nn.Module):
    """GSConv: standard conv (half channels) + depthwise 5×5 + channel shuffle."""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)  # depthwise 5x5

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # channel shuffle
        b, n, h, w = x2.size()
        x2 = x2.view(b, 2, n // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, n, h, w)
        return x2


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """Drop-in replacement for C3k2 in the neck."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((x1, y), 1))