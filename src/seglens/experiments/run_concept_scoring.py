"""E4: scoring de conceptos clínicos por etapa.

Genera mapas de alineación concepto-por-concepto sobre el test set y los
resumen como CSV/heatmap.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch.utils.data import DataLoader

from seglens.adapter import SegmentationAdapter
from seglens.concepts import (
    ConceptThresholds,
    concept_scores,
    derive_concept_masks,
    fit_cavs,
)
from seglens.hooks import FeatureExtractor


@torch.no_grad()
def fit_concepts(
    adapter: SegmentationAdapter,
    calibration_loader: DataLoader,
    n_images: int = 200,
    device: str = "cpu",
) -> tuple[dict[str, dict[str, torch.Tensor]], ConceptThresholds]:
    """Calcula umbrales T1c y ajusta CAVs por etapa.

    Devuelve ``cavs: {stage: {concept: cav}}`` y los umbrales.
    """
    adapter.model.to(device).eval()
    # 1) umbrales
    seen = 0
    fg_bg_imgs, fg_bg_masks = [], []
    for batch in calibration_loader:
        if seen >= n_images:
            break
        x, m = batch[0], batch[1]
        take = min(x.shape[0], n_images - seen)
        fg_bg_imgs.append(x[:take])
        fg_bg_masks.append(m[:take])
        seen += take
    images = torch.cat(fg_bg_imgs, dim=0).to(device)
    masks = torch.cat(fg_bg_masks, dim=0).to(device).float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    thresholds = ConceptThresholds.from_batch(images, masks, t1c_channel=1)

    # 2) CAVs por etapa
    cavs: dict[str, dict[str, torch.Tensor]] = {}
    seen = 0
    for batch in calibration_loader:
        if seen >= n_images:
            break
        x, m = batch[0], batch[1]
        x = adapter.preprocess(x).to(device)
        m = m.to(device).float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        b = x.shape[0]
        take = min(b, n_images - seen)
        if take < b:
            x, m = x[:take], m[:take]
        with FeatureExtractor(adapter.model, adapter.stages) as ex:
            _ = adapter.segment(x)
        feats = ex.features
        concept_masks = derive_concept_masks(x, m, thresholds, t1c_channel=1)
        for stage in adapter.stages:
            cavs_stage = fit_cavs(feats, concept_masks, stage)
            if stage not in cavs:
                cavs[stage] = cavs_stage
        seen += take
    return cavs, thresholds


@torch.no_grad()
def score_concepts_on_test(
    adapter: SegmentationAdapter,
    cavs: dict[str, dict[str, torch.Tensor]],
    test_loader: DataLoader,
    device: str = "cpu",
    max_images: int | None = None,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    """Para cada imagen de test, devuelve los mapas de score por etapa y concepto.

    Salida: ``{stage: {concept: [tensor_per_image]}}`` donde cada tensor es
    ``(1, H, W)`` con los scores.
    """
    adapter.model.to(device).eval()
    out: dict[str, dict[str, list[torch.Tensor]]] = {
        s: {c: [] for c in next(iter(cavs.values())).keys()} for s in adapter.stages
    }
    seen = 0
    for batch in test_loader:
        if max_images is not None and seen >= max_images:
            break
        x = adapter.preprocess(batch[0]).to(device)
        with FeatureExtractor(adapter.model, adapter.stages) as ex:
            _ = adapter.segment(x)
        feats = ex.features
        for stage in adapter.stages:
            scores = concept_scores(feats[stage], cavs[stage])
            for concept, s_map in scores.items():
                # guardar solo primera imagen del batch para mantenerlo manejable
                out[stage][concept].append(s_map.cpu())
        seen += x.shape[0]
    return out
