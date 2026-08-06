"""Métricas de evaluación: dice, IoU, ECE ligero."""

from __future__ import annotations

import torch
from torch import Tensor


def dice_score(pred: Tensor, target: Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """Dice binario entre logits/probabilidades y target binario."""
    p = (torch.sigmoid(pred) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum().item()
    denom = p.sum().item() + t.sum().item()
    if denom < eps:
        return 1.0
    return 2.0 * inter / denom


def iou_score(pred: Tensor, target: Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """IoU binario entre logits/probabilidades y target binario."""
    p = (torch.sigmoid(pred) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum().item()
    union = p.sum().item() + t.sum().item() - inter
    if union < eps:
        return 1.0
    return inter / union
