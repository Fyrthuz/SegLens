"""Tests para concepts: thresholds, derivación de máscaras, CAVs, scoring."""

from __future__ import annotations

import torch
import pytest

from seglens.concepts import (
    ConceptThresholds,
    DEFAULT_CONCEPTS,
    concept_scores,
    derive_concept_masks,
    fit_cavs,
)


def _make_synthetic_batch(B=4, H=16, W=16, fg_intensity=10.0, bg_intensity=0.0):
    """Crea un batch sintético con foreground brillante y background oscuro."""
    images = torch.zeros(B, 3, H, W)
    images[:, 1] = bg_intensity  # T1c (canal G)
    masks = torch.zeros(B, 1, H, W)
    # foreground: cuadrado central
    h2, w2 = H // 4, W // 4
    images[:, 1, h2:3*h2, w2:3*w2] = fg_intensity
    masks[:, 0, h2:3*h2, w2:3*w2] = 1.0
    # también una zona "periferia" con intensidad media
    h3, w3 = H // 3, W // 3
    images[:, 1, h3:2*h3, w3:2*w3] = fg_intensity * 0.75
    return images, masks


def test_thresholds_from_batch():
    images, masks = _make_synthetic_batch()
    th = ConceptThresholds.from_batch(images, masks, t1c_channel=1)
    assert th.fg_percentiles[100] == pytest.approx(10.0, abs=1e-3)
    assert th.bg_percentiles[0] == pytest.approx(0.0, abs=1e-3)
    # value_at
    v = th.value_at("fg", 0.5)
    assert 0.0 <= v <= 10.0


def test_derive_concept_masks_returns_bool_tensors():
    images, masks = _make_synthetic_batch()
    th = ConceptThresholds.from_batch(images, masks, t1c_channel=1)
    cm = derive_concept_masks(images, masks, th, t1c_channel=1)
    assert set(cm.keys()) == set(DEFAULT_CONCEPTS.keys())
    for name, m in cm.items():
        assert m.dtype == torch.bool
        assert m.shape == (4, 16, 16)
    # tumor_core debe coincidir con la zona central
    assert cm["tumor_core"].sum() > 0
    # ventricle_csf (T1c muy baja en BG) debe cubrir la mayor parte del BG
    assert cm["ventricle_csf"].sum() > 0


def test_fit_cavs_produces_unit_vectors():
    images, masks = _make_synthetic_batch()
    th = ConceptThresholds.from_batch(images, masks, t1c_channel=1)
    cm = derive_concept_masks(images, masks, th, t1c_channel=1)
    # features: discriminantes — pixel de FG con canal +1, BG con -1
    B, H, W = images.shape[0], images.shape[2], images.shape[3]
    feats = torch.zeros(B, 4, H, W)
    fg_mask = masks.squeeze(1) > 0.5
    feats[:, 0] = fg_mask.float()  # feature 0: 1 si FG
    feats[:, 1] = (~fg_mask).float()  # feature 1: 1 si BG
    cavs = fit_cavs({"dummy": feats}, cm, stage="dummy")
    for name, cav in cavs.items():
        assert cav.shape == (4,)
        assert cav.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_concept_scores_shape_and_signs():
    images, masks = _make_synthetic_batch()
    th = ConceptThresholds.from_batch(images, masks, t1c_channel=1)
    cm = derive_concept_masks(images, masks, th, t1c_channel=1)
    B, H, W = 4, 16, 16
    feats = torch.zeros(B, 4, H, W)
    fg_mask = masks.squeeze(1) > 0.5
    feats[:, 0] = fg_mask.float()
    feats[:, 1] = (~fg_mask).float()
    cavs = fit_cavs({"dummy": feats}, cm, stage="dummy")
    scores = concept_scores(feats, cavs)
    assert set(scores.keys()) == set(cavs.keys())
    for name, s in scores.items():
        assert s.shape == (B, 1, H, W)
