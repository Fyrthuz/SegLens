"""Tests para la lente 3: JacobianLens (J-lens analógico).

Estos tests usan un modelo minúsculo y verifican propiedades
matemáticas clave: causalidad de primer orden, recuperación de la cabeza
en la última capa (aproximadamente), shape de la Jacobiana.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from seglens.adapter import UNetAdapter
from seglens.lens.jacobian_lens import JacobianLens
from seglens.third_party import UNet


class LinearCNN(nn.Module):
    """Modelo totalmente lineal (sin ReLU/pool) para hacer Jacobianos exactos."""

    def __init__(self, c1=4, c2=8, c3=1):
        super().__init__()
        self.body = nn.Conv2d(c1, c2, 1)  # 1x1 = pixelwise
        self.head = nn.Conv2d(c2, c3, 1)
        self._stages = ["body", "head"]

    def forward(self, x):
        return self.head(self.body(x))


def _make_dummy_loader(adapter, n=4, h=8, w=8, c=3):
    """Genera batches (x, y) aleatorios sin etiqueta real."""
    def loader():
        for _ in range(n):
            x = torch.randn(1, c, h, w)
            y = torch.randint(0, 2, (1, 1, h, w)).float()
            yield (x, y)
    return loader()


def test_calibrate_runs_and_produces_jacobians():
    """Smoke test: la calibración produce una Jacobiana por stage."""
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    adapter = UNetAdapter(m).eval()
    lens = JacobianLens(target_hw=(16, 16), out_channels=1)
    loader = _make_dummy_loader(adapter, n=2, h=16, w=16)
    lens.calibrate(adapter, loader, n_images=2, device="cpu")
    # Al menos un stage debe tener Jacobiano
    assert len(lens.stages) > 0
    for stage in lens.stages:
        J = lens.jacobian_for(stage)
        assert J.dim() == 2
        assert J.shape[0] == 1  # out_channels
        # la segunda dimensión es C_l de la etapa
        assert J.shape[1] > 0


def test_jacobian_for_linear_cnn_recovers_head():
    """En un modelo totalmente lineal, la Jacobiana de la feature de body
    (justo antes de la cabeza) debería aproximar el peso de la cabeza."""
    m = LinearCNN(c1=4, c2=8, c3=1)
    # Construimos el adapter con los stages correctos para LinearCNN
    adapter = UNetAdapter(m, stages=["body", "head"])

    lens = JacobianLens(target_hw=(8, 8), out_channels=1)
    loader = _make_dummy_loader(adapter, n=2, h=8, w=8, c=4)
    lens.calibrate(adapter, loader, n_images=2, device="cpu", max_pixels_per_image=None)

    # La Jacobiana en "body" debería parecerse al peso de la cabeza (1, 8) → (1, 1)
    J_body = lens.jacobian_for("body")  # (1, C_l_body) = (1, 8)
    W_head = m.head.weight.detach()  # (1, 8, 1, 1)
    assert J_body.shape == (1, 8), f"J_body shape: {J_body.shape}"
    # muy próximos porque head es lineal pixelwise
    err = (J_body - W_head.view(1, 8)).abs().max().item()
    assert err < 1e-3, f"Error demasiado alto: {err}"


def test_call_raises_for_uncalibrated_stage():
    lens = JacobianLens(target_hw=(8, 8))
    with pytest.raises(KeyError):
        lens("nope", torch.randn(1, 4, 8, 8))


def test_call_produces_correct_shape():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    adapter = UNetAdapter(m).eval()
    lens = JacobianLens(target_hw=(16, 16), out_channels=1)
    loader = _make_dummy_loader(adapter, n=2, h=16, w=16)
    lens.calibrate(adapter, loader, n_images=2, device="cpu")
    feats = torch.randn(1, 8, 16, 16)  # canales del primer stage
    # Buscar un stage que tenga 8 canales
    target_stage = None
    for s in lens.stages:
        J = lens.jacobian_for(s)
        if J.shape[1] == 8:
            target_stage = s
            break
    if target_stage is None:
        pytest.skip("Ninguna etapa calibrada con 8 canales (configuración cambiada)")
    out = lens(target_stage, feats)
    assert out.logits.shape == (1, 1, 16, 16)


def test_save_and_load():
    m = UNet(in_channels=3, out_channels=1, init_features=8)
    adapter = UNetAdapter(m).eval()
    lens = JacobianLens(target_hw=(16, 16), out_channels=1)
    loader = _make_dummy_loader(adapter, n=2, h=16, w=16)
    lens.calibrate(adapter, loader, n_images=2, device="cpu")
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "jac.pt")
        lens.save(p)
        lens2 = JacobianLens(target_hw=(16, 16), out_channels=1).load(p)
    for stage in lens.stages:
        assert stage in lens2.stages
        assert torch.allclose(lens.jacobian_for(stage), lens2.jacobian_for(stage), atol=1e-5)
