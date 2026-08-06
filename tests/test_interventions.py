"""Tests para interventions: ablate, steer, patch, ablate_concept."""

from __future__ import annotations

import torch
import pytest

from seglens.interventions import ablate_concept, ablate_direction, patch_activations, steer


def test_ablate_direction_removes_projection():
    torch.manual_seed(0)
    features = torch.randn(2, 4, 8, 8)
    direction = torch.randn(4)
    out = ablate_direction(features, direction)
    # Tras la ablación, <out, direction> debe ser ~0
    flat = out.permute(0, 2, 3, 1).reshape(-1, 4)
    d = direction / direction.norm()
    coeff = flat @ d
    assert coeff.abs().max().item() < 1e-5


def test_ablate_direction_preserves_orthogonal_components():
    torch.manual_seed(0)
    features = torch.randn(1, 4, 4, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = ablate_direction(features, direction)
    # La feature 0 debe ser 0, las demás intactas
    assert torch.allclose(out[0, 0], torch.zeros(4, 4), atol=1e-6)
    assert torch.allclose(out[0, 1:], features[0, 1:])


def test_steer_adds_alpha_times_direction():
    features = torch.zeros(1, 3, 4, 4)
    direction = torch.tensor([1.0, 2.0, 3.0])
    out = steer(features, direction, alpha=2.0)
    expected = torch.full((1, 3, 4, 4), 0.0)
    expected[0, 0] = 2.0
    expected[0, 1] = 4.0
    expected[0, 2] = 6.0
    assert torch.allclose(out, expected)


def test_patch_activations_with_mask():
    src = torch.full((1, 3, 4, 4), 9.0)
    tgt = torch.full((1, 3, 4, 4), 1.0)
    mask = torch.zeros(1, 4, 4)
    mask[0, 1:3, 1:3] = 1.0
    out = patch_activations(tgt, src, mask)
    # fuera de la máscara: tgt=1; dentro: src=9
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[0, 0, 2, 2].item() == 9.0


def test_patch_activations_upsamples_mask():
    src = torch.full((1, 1, 8, 8), 5.0)
    tgt = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 1:3, 1:3] = 1.0
    out = patch_activations(tgt, src, mask)
    # La máscara upsampled nearest debe copiar 5 en el centro 4x4
    assert (out[0, 0, 2:6, 2:6] == 5.0).all()
    assert (out[0, 0, 0:2, :] == 0.0).all()


def test_ablate_concept_partial_strength():
    torch.manual_seed(0)
    features = torch.randn(1, 4, 4, 4)
    cav = torch.randn(4)
    full = ablate_concept(features, cav, strength=1.0)
    none = ablate_concept(features, cav, strength=0.0)
    assert torch.allclose(none, features)
    # full ablation: proyección ~0
    flat = full.permute(0, 2, 3, 1).reshape(-1, 4)
    d = cav / cav.norm()
    assert (flat @ d).abs().max().item() < 1e-5
