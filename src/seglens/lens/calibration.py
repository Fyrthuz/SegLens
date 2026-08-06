"""Orquestador: calibra las tres lentes sobre un dataloader común."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from seglens.adapter import SegmentationAdapter
from seglens.hooks import FeatureExtractor
from seglens.lens.jacobian_lens import JacobianLens
from seglens.lens.logit_lens import LogitLens
from seglens.lens.tuned_lens import TunedLens


def _collect_stage_features(
    adapter: SegmentationAdapter, x: Tensor
) -> tuple[dict[str, Tensor], Tensor]:
    """Pasa forward capturando features y devolviendo el target (logits)."""
    x = adapter.preprocess(x)
    with FeatureExtractor(adapter.model, adapter.stages) as ex:
        with torch.no_grad():
            y = adapter.model(x)
    return ex.features, y.detach()


def _features_to_pixels(features: dict[str, Tensor], target_hw: tuple[int, int]) -> dict[str, Tensor]:
    """Reshapea features a ``(N_pixels, C_l)`` bilinearly-upsampleadas a target_hw."""
    import torch.nn.functional as F

    out: dict[str, Tensor] = {}
    for stage, f in features.items():
        if tuple(f.shape[-2:]) != target_hw:
            f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
        B, C, H, W = f.shape
        out[stage] = f.permute(0, 2, 3, 1).reshape(-1, C).detach()
    return out


def run_calibration(
    adapter: SegmentationAdapter,
    dataloader: DataLoader,
    n_images: int = 200,
    device: str = "cpu",
):
    """Calibra las 3 lentes y devuelve ``{"logit", "tuned", "jacobian"}``."""
    adapter.model.to(device).eval()

    # Determinar target_hw con la primera imagen
    first = next(iter(dataloader))
    x0 = first[0] if isinstance(first, (list, tuple)) else first
    x0 = adapter.preprocess(x0[:1]).to(device)
    with torch.no_grad():
        y0 = adapter.model(x0)
    target_hw = tuple(y0.shape[-2:])
    out_channels = y0.shape[1]

    # 1) Logit lens
    logit = LogitLens(adapter.head(), target_hw=target_hw)

    # 2) Tuned lens: acumula features y targets píxel a píxel
    tuned = TunedLens(target_hw=target_hw, out_channels=out_channels)
    accum_features: dict[str, list[Tensor]] = {s: [] for s in adapter.stages}
    accum_targets: list[Tensor] = []
    seen = 0
    for batch in dataloader:
        if seen >= n_images:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = adapter.preprocess(x).to(device)
        b = x.shape[0]
        take = min(b, n_images - seen)
        if take < b:
            x = x[:take]
        feats, y = _collect_stage_features(adapter, x)
        flat_feats = _features_to_pixels(feats, target_hw)
        for s in adapter.stages:
            accum_features[s].append(flat_feats[s])
        y_up = y
        if tuple(y.shape[-2:]) != target_hw:
            import torch.nn.functional as F

            y_up = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        B, K, H, W = y_up.shape
        accum_targets.append(y_up.permute(0, 2, 3, 1).reshape(-1, K).detach())
        seen += take
    for s in adapter.stages:
        if accum_features[s]:
            tuned.fit({s: torch.cat(accum_features[s], dim=0)}, torch.cat(accum_targets, dim=0))

    # 3) Jacobian lens
    jacobian = JacobianLens(target_hw=target_hw, out_channels=out_channels)
    jacobian.calibrate(adapter, dataloader, n_images=n_images, device=device)

    return {"logit": logit, "tuned": tuned, "jacobian": jacobian}
