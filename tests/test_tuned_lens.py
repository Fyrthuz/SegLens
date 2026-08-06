"""Tests para la lente 2: TunedLens (probe lineal por mínimos cuadrados)."""

from __future__ import annotations

import torch
import pytest

from seglens.lens.tuned_lens import TunedLens


def test_fit_and_predict_perfectly_on_simple_task():
    """Si la feature es una proyección lineal exacta del target, la probe debe
    recuperar la transformación."""
    torch.manual_seed(0)
    N, C, K = 500, 16, 1
    true_W = torch.randn(C, K)
    X = torch.randn(N, C)
    Y = X @ true_W + 0.01 * torch.randn(N, K)
    lens = TunedLens(target_hw=(8, 8), out_channels=K)
    lens.fit({"stage_a": X}, Y, l2=1e-6)
    # El probe está en lens._probes["stage_a"].weight → (K, C)
    W = lens._probes["stage_a"].weight  # (K, C)
    # Comparamos con true_W.T
    err = (W - true_W.T).norm() / true_W.norm()
    assert err < 0.05  # muy buena recuperación


def test_fit_fails_on_shape_mismatch():
    lens = TunedLens(target_hw=(8, 8))
    with pytest.raises(ValueError, match="2D"):
        lens.fit({"a": torch.randn(10, 4, 4)}, torch.randn(10, 1))


def test_fit_fails_on_pixel_mismatch():
    lens = TunedLens(target_hw=(8, 8))
    with pytest.raises(ValueError, match="Nº de píxeles"):
        lens.fit({"a": torch.randn(10, 4)}, torch.randn(11, 1))


def test_call_raises_for_unfitted_stage():
    lens = TunedLens(target_hw=(8, 8))
    with pytest.raises(KeyError, match="sin probe"):
        lens("never_fitted", torch.randn(1, 4, 8, 8))


def test_call_upsamples_and_produces_logits():
    torch.manual_seed(0)
    lens = TunedLens(target_hw=(16, 16), out_channels=1)
    X = torch.randn(64, 4)
    Y = X @ torch.randn(4, 1)
    lens.fit({"s": X}, Y, l2=1e-4)
    feats = torch.randn(1, 4, 8, 8)
    out = lens("s", feats)
    assert out.logits.shape == (1, 1, 16, 16)
