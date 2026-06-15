"""Focaler-CIoU loss for hard-sample-focused bounding box regression.
Reference: Zhang et al., "Focaler-IoU: More Focused Intersection over Union Loss" (2024).
Used in FracDet-v11 (Sci Reports 2026) for pediatric wrist fracture detection.
"""
import math
import torch


def focaler_ciou(box1, box2, d=0.0, u=0.95, eps=1e-7):
    """Focaler-CIoU between box1 and box2. Both in xyxy, shape [..., 4]. Returns [..., 1]."""
    # Coordinates
    b1 = box1.chunk(4, -1)
    b2 = box2.chunk(4, -1)
    x1, y1, x2, y2 = b1[0], b1[1], b1[2], b1[3]
    x1g, y1g, x2g, y2g = b2[0], b2[1], b2[2], b2[3]

    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x2g - x1g, y2g - y1g

    # IoU
    inter = (torch.min(x2, x2g) - torch.max(x1, x1g)).clamp(min=0) * \
            (torch.min(y2, y2g) - torch.max(y1, y1g)).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Focaler mapping: linear interval [d, u] → [0, 1]
    focaler_iou = ((iou - d) / (u - d)).clamp(0.0, 1.0)

    # CIoU penalty (enclosing box + aspect ratio)
    cw = torch.max(x2, x2g) - torch.min(x1, x1g)
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((x1g + x2g - x1 - x2).pow(2) + (y1g + y2g - y1 - y2).pow(2)) / 4

    v = (4 / math.pi ** 2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return focaler_iou - (rho2 / c2 + v * alpha)