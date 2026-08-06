"""E1 + E2: calibración de lentes y curvas de profundidad de decisión."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from seglens.adapter import SegmentationAdapter
from seglens.hooks import FeatureExtractor
from seglens.lens import JacobianLens, LogitLens, TunedLens
from seglens.lens.base import LensOutput
from seglens.metrics import dice_score, iou_score


def run_depth_curves(
    adapter: SegmentationAdapter,
    lenses: Mapping[str, LogitLens | TunedLens | JacobianLens],
    dataloader: DataLoader,
    device: str = "cpu",
    max_images: int | None = None,
):
    """Itera el dataloader, aplica cada lente en cada etapa, mide Dice/IoU.

    Devuelve ``per_image: list[dict[lens_name, dict[stage, LensOutput|None]]]``
    y ``targets: list[Tensor]``.
    """
    adapter.model.to(device).eval()
    per_image: list[dict[str, dict[str, LensOutput | None]]] = []
    targets: list[Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            x = adapter.preprocess(x).to(device)
            y = y.to(device).float()
            if y.dim() == 3:
                y = y.unsqueeze(1)
            with FeatureExtractor(adapter.model, adapter.stages) as ex:
                _ = adapter.model(x)
            feats = ex.features
            img_results: dict[str, dict[str, LensOutput | None]] = {}
            for lens_name, lens in lenses.items():
                img_results[lens_name] = {}
                for stage in adapter.stages:
                    if hasattr(lens, "has") and not lens.has(stage):
                        img_results[lens_name][stage] = None
                        continue
                    out = lens(stage, feats[stage].to(device))
                    img_results[lens_name][stage] = out
            per_image.append(img_results)
            for b in range(x.shape[0]):
                targets.append(y[b])
            seen += x.shape[0]
            if max_images is not None and seen >= max_images:
                break
    return per_image, targets


def summarize_depth_curves(per_image, targets) -> dict:
    """Promedia Dice/IoU por ``lente -> stage``."""
    from seglens.viz import aggregate_depth_curve

    return aggregate_depth_curve(per_image, targets)


def save_results(aggregated, csv_path: str | Path, plot_path: str | Path | None = None):
    from seglens.viz import plot_depth_curves, save_depth_curve_csv

    save_depth_curve_csv(aggregated, csv_path)
    if plot_path is not None:
        plot_depth_curves(aggregated, save_path=plot_path)
