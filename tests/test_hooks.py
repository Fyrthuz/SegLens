"""Tests para FeatureExtractor."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from seglens.hooks import FeatureExtractor, resolve_spatial


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(16 * 4 * 4, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


def test_captures_requested_stages():
    m = TinyCNN()
    x = torch.randn(2, 3, 16, 16)
    with FeatureExtractor(m, ["conv1", "conv2", "pool"]) as ex:
        out = m(x)
    assert set(ex.features.keys()) == {"conv1", "conv2", "pool"}
    assert ex.features["conv1"].shape == (2, 8, 16, 16)
    assert ex.features["conv2"].shape == (2, 16, 16, 16)
    assert ex.features["pool"].shape == (2, 16, 4, 4)
    assert out.shape == (2, 1)


def test_invalid_stage_raises():
    m = TinyCNN()
    with pytest.raises(ValueError):
        FeatureExtractor(m, ["conv1", "nope"])


def test_hooks_removed_after_context():
    m = TinyCNN()
    x = torch.randn(1, 3, 16, 16)
    with FeatureExtractor(m, ["conv1"]) as ex:
        m(x)
    assert ex.features != {}
    # fuera del contexto no quedan hooks residuales
    with FeatureExtractor(m, ["conv1"]) as ex2:
        m(x)
    assert set(ex2.features.keys()) == {"conv1"}


def test_resolve_spatial_upsamples():
    feats = {
        "low": torch.randn(1, 4, 8, 8),
        "high": torch.randn(1, 4, 32, 32),
    }
    target = (16, 16)
    out = resolve_spatial(feats, target)
    assert out["low"].shape == (1, 4, 16, 16)
    assert out["high"].shape == (1, 4, 16, 16)
    # ya en target → no se modifica
    feats_same = {"x": torch.randn(1, 4, 16, 16)}
    out_same = resolve_spatial(feats_same, (16, 16))
    assert torch.equal(out_same["x"], feats_same["x"])


def test_resolve_spatial_rejects_non_4d():
    feats = {"bad": torch.randn(1, 4, 8)}
    with pytest.raises(ValueError, match="no es 4D"):
        resolve_spatial(feats, (8, 8))
