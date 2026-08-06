"""Tests para métricas y utilidades de visualización."""

from __future__ import annotations

import torch
import pytest

from seglens.metrics import dice_score, iou_score


def test_dice_perfect():
    p = torch.tensor([1.0, 0.0, 1.0, 0.0])
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert dice_score(p, t) == 1.0


def test_dice_zero():
    p = torch.tensor([1.0, 1.0, 1.0, 1.0])
    t = torch.zeros(4)
    assert dice_score(p, t) == 0.0


def test_dice_half():
    """Pred=4 positivos, GT=4 positivos, intersección=2, Dice = 2*2/(4+4) = 0.5."""
    p = torch.tensor([1.0, 1.0, 0.0, 0.0])
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # sigmoid(5) > 0.5 → todo 1; sigmoid(0) → 0.5 (umbral >0.5 → 0). Usemos valores claros.
    p_logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
    t_bin = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # inter = 1 (p[0]=t[0]=1), union_p = 2, union_t = 2
    # Dice = 2*1 / (2+2) = 0.5
    assert dice_score(p_logits, t_bin) == pytest.approx(0.5)


def test_iou():
    p = torch.tensor([5.0, 5.0, -5.0, -5.0])
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # inter = 1, union = 2+2-1 = 3
    assert iou_score(p, t) == pytest.approx(1 / 3)


def test_iou_perfect():
    p = torch.tensor([5.0, -5.0, 5.0, -5.0])
    t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert iou_score(p, t) == 1.0


def test_dice_handles_empty():
    """Si no hay positivos en pred ni en target, devolvemos 1.0 (convención)."""
    p = torch.full((4,), -10.0)
    t = torch.zeros(4)
    assert dice_score(p, t) == 1.0
    assert iou_score(p, t) == 1.0
