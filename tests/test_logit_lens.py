"""Tests para la lente 1: LogitLens."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import pytest

from seglens.lens.logit_lens import LogitLens


class Conv1x1Model(nn.Module):
    """Modelo trivial con dos stages (intermedia + cabeza 1x1)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        return self.head(torch.relu(self.body(x)))


def test_compatible_with_same_channels():
    m = Conv1x1Model()
    lens = LogitLens(m.head, target_hw=(16, 16))
    feats = torch.randn(2, 8, 16, 16)
    out = lens("intermediate", feats)
    assert out is not None
    assert out.logits.shape == (2, 1, 16, 16)
    assert out.stage == "intermediate"
    assert out.lens == "logit"


def test_incompatible_channels_returns_none():
    m = Conv1x1Model()
    lens = LogitLens(m.head, target_hw=(16, 16))
    feats = torch.randn(2, 16, 16, 16)  # 16 ≠ head.in_channels
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = lens("intermediate", feats)
    assert out is None
    assert any("LogitLens" in str(x.message) for x in w)


def test_upsamples_to_target_hw():
    m = Conv1x1Model()
    lens = LogitLens(m.head, target_hw=(32, 32))
    feats = torch.randn(1, 8, 16, 16)
    out = lens("intermediate", feats)
    assert out.logits.shape == (1, 1, 32, 32)


def test_per_image_helper():
    m = Conv1x1Model()
    lens = LogitLens(m.head, target_hw=(16, 16))
    out = lens.per_image("x", torch.randn(1, 8, 16, 16))
    assert out.shape == (1, 1, 16, 16)
    assert lens.per_image("x", torch.randn(1, 16, 16, 16)) is None
