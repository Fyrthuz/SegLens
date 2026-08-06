"""Visualización: curvas de profundidad, mapas cualitativos, heatmaps de conceptos.

Nota: estas funciones no importan ``matplotlib`` arriba para mantener
imports ligeros en tests; cada función lo carga perezosamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from seglens.lens.base import LensOutput
from seglens.metrics import dice_score, iou_score


def aggregate_depth_curve(
    per_image: Sequence[Mapping[str, Mapping[str, LensOutput]]],
    targets: Sequence[Tensor],
) -> dict[str, dict[str, list[float]]]:
    """Agrega Dice/IoU por ``lente -> stage -> lista_por_imagen``.

    Parameters
    ----------
    per_image : lista por imagen de ``{lente: {stage: LensOutput}}``
    targets : lista por imagen de ``(1, H, W)`` máscaras ground-truth.
    """
    from seglens.lens.base import LensOutput as _L

    out: dict[str, dict[str, list[float]]] = {}
    for img_idx, lens_map in enumerate(per_image):
        target = targets[img_idx]
        for lens_name, stage_map in lens_map.items():
            for stage, lout in stage_map.items():
                if lout is None:
                    continue
                d = dice_score(lout.logits, target)
                i = iou_score(lout.logits, target)
                out.setdefault(lens_name, {}).setdefault(stage, {"dice": [], "iou": []})
                out[lens_name][stage]["dice"].append(d)
                out[lens_name][stage]["iou"].append(i)
    return out


def save_depth_curve_csv(
    aggregated: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    path: str | Path,
) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lens", "stage", "n_images", "dice_mean", "dice_std", "iou_mean", "iou_std"])
        for lens_name, stage_map in aggregated.items():
            for stage, metrics in stage_map.items():
                ds = metrics.get("dice", [])
                iss = metrics.get("iou", [])
                if not ds:
                    continue
                w.writerow([
                    lens_name,
                    stage,
                    len(ds),
                    f"{sum(ds)/len(ds):.4f}",
                    f"{torch.tensor(ds).std().item():.4f}",
                    f"{sum(iss)/len(iss):.4f}",
                    f"{torch.tensor(iss).std().item():.4f}",
                ])


def plot_depth_curves(
    aggregated: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    stages: Sequence[str] | None = None,
    metric: str = "dice",
    title: str = "Profundidad de decisión",
    save_path: str | Path | None = None,
):
    """Dibuja la curva ``metric-vs-stage`` para cada lente."""
    import matplotlib.pyplot as plt

    if stages is None:
        # tomar la unión ordenada
        s: set[str] = set()
        for stage_map in aggregated.values():
            s.update(stage_map.keys())
        stages = list(s)
        # intentar ordenarlas por nº de canales ascendente (heurística: orden de aparición)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"logit": "tab:gray", "tuned": "tab:orange", "jacobian": "tab:blue"}
    for lens_name, stage_map in aggregated.items():
        xs, ys, stds = [], [], []
        for i, stage in enumerate(stages):
            ms = stage_map.get(stage)
            if not ms:
                continue
            vals = ms.get(metric, [])
            if not vals:
                continue
            xs.append(i)
            ys.append(sum(vals) / len(vals))
            stds.append(torch.tensor(vals).std().item())
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=stds,
                label=lens_name,
                marker="o",
                color=colors.get(lens_name, None),
                capsize=2,
            )
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Etapa")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
