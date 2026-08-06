"""Tests para uncertainty.py."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytest

from seglens.adapter import UNetAdapter, UniVerSegAdapter
from seglens.third_party import UNet
from seglens.uncertainty import (
    MCDropoutInjector,
    UncertaintyEstimator,
    _default_mc_stages,
    _tta_transforms_unet,
    _tta_transforms_universeg,
)


# ----- MCDropoutInjector -----


def test_injector_rejects_invalid_p():
    m = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="p debe estar"):
        MCDropoutInjector(m, stages=[], p=0.0)


def test_injector_rejects_invalid_stage():
    m = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="stages inexistentes"):
        MCDropoutInjector(m, stages=["nonexistent"], p=0.5)


def test_injector_modifies_forward_output():
    """El injector debe modificar la salida del forward cuando está activo."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Conv2d(3, 8, 3, padding=1)
            self.head = nn.Conv2d(8, 1, 1)

        def forward(self, x):
            return self.head(torch.relu(self.body(x)))

    m = TinyModel().eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        y_clean = m(x)
    with MCDropoutInjector(m, stages=["body"], p=0.5):
        with torch.no_grad():
            y_drop1 = m(x)
            y_drop2 = m(x)
    # Las salidas dropeadas deben diferir entre sí y del clean
    assert not torch.allclose(y_clean, y_drop1, atol=1e-6)
    assert not torch.allclose(y_drop1, y_drop2, atol=1e-6)


def test_injector_hooks_removed_after_context():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.body(x)

    m = TinyModel().eval()
    x = torch.randn(1, 3, 8, 8)
    with MCDropoutInjector(m, stages=["body"], p=0.5):
        m(x)
    # fuera del contexto, no hay hooks
    assert len(m._forward_hooks) == 0 or all(
        h is None for h in m._forward_hooks.values()
    )


# ----- TTA transforms -----


def test_tta_unet_has_six_with_rotation():
    ts = _tta_transforms_unet(with_rotation=True)
    names = [t[0] for t in ts]
    assert names == ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]


def test_tta_unet_no_rotation():
    ts = _tta_transforms_unet(with_rotation=False)
    names = [t[0] for t in ts]
    assert names == ["identity", "hflip", "vflip"]


def test_tta_universeg_only_hflip():
    ts = _tta_transforms_universeg(with_rotation=True)
    names = [t[0] for t in ts]
    assert names == ["identity", "hflip"]


def test_tta_hflip_is_invertible():
    x = torch.randn(1, 3, 16, 16)
    for _, x_fn, y_inv in _tta_transforms_unet():
        y = y_inv(x_fn(x))
        assert torch.allclose(x, y, atol=1e-6), f"transform no invertible"


# ----- _default_mc_stages -----


def test_default_mc_stages_unet():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m)
    assert _default_mc_stages(a) == ["up3", "up4"]


def test_default_mc_stages_universeg():
    """Mock: no requiere universeg instalado."""
    class FakeUni(nn.Module):
        def forward(self, x, ci, cm):
            return x
    a = UniVerSegAdapter(FakeUni())
    stages = _default_mc_stages(a)
    assert "dec_blocks.2.target.vmapped" in stages


# ----- UncertaintyEstimator -----


def test_estimator_noisy_adds_variance():
    """N passes con noisy deben dar predicciones distintas."""
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=5, noise_sigma=0.05, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    out = est.noisy(x)
    assert "mean" in out and "std" in out and "entropy" in out
    assert out["mean"].shape == (1, 1, 32, 32)
    assert out["std"].shape == (1, 1, 32, 32)
    assert out["entropy"].shape == (1, 1, 32, 32)
    # std debe ser > 0 con sigma=0.05
    assert out["std"].mean().item() > 0


def test_estimator_noisy_with_zero_sigma_collapse():
    """Con sigma=0, todas las predicciones son iguales → std=0."""
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=5, noise_sigma=0.0, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    out = est.noisy(x)
    assert out["std"].max().item() < 1e-5


def test_estimator_mc_dropout_with_injector():
    """Si el modelo no tiene dropout, el injector añade varianza vía hook."""
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, mc_stages=["up3", "up4"], mc_p=0.5,
                                n_samples=5, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    out = est.mc_dropout(x)
    assert out["std"].mean().item() > 0


def test_estimator_uncertainty_map_dispatch():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=3, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    out_mc = est.uncertainty_map(x, method="mc_dropout")
    out_tta = est.uncertainty_map(x, method="tta")
    out_n = est.uncertainty_map(x, method="noisy")
    assert all(k in out_mc for k in ("mean", "std", "entropy"))
    assert all(k in out_tta for k in ("mean", "std", "entropy"))
    assert all(k in out_n for k in ("mean", "std", "entropy"))


def test_estimator_unknown_method_raises():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=3, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="method debe ser"):
        est.uncertainty_map(x, method="foo")


def test_estimator_entropy_in_range():
    """Entropy ∈ [0, log 2] para predicción binaria."""
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=3, noise_sigma=0.05, device="cpu")
    x = torch.randn(1, 3, 32, 32)
    out = est.noisy(x)
    log2 = math.log(2)
    assert (out["entropy"] >= 0).all()
    assert (out["entropy"] <= log2 + 1e-5).all()


def test_estimator_calibrate_baseline():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    a = UNetAdapter(m).eval()
    est = UncertaintyEstimator(a, n_samples=3, device="cpu")
    # Mini dataloader
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(4, 3, 16, 16), torch.zeros(4, 1, 16, 16))
    loader = DataLoader(ds, batch_size=1)
    baseline = est.calibrate_baseline(loader, n=3)
    assert "mean_entropy" in baseline
    assert baseline["n_images"] == 3
