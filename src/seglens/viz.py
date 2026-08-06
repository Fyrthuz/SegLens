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
    uncertainty_per_image_stage: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Agrega Dice/IoU por ``lente -> stage -> lista_por_imagen``.

    Parámetros
    ----------
    uncertainty_per_image_stage : ``{lens: {stage: [std_per_image]}}`` opcional.
        Si se pasa, añade a cada entry un campo ``dice_uncertainty`` con la
        lista de stds de Dice bajo N pases MC/TTA por imagen.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for img_idx, lens_map in enumerate(per_image):
        target = targets[img_idx]
        for lens_name, stage_map in lens_map.items():
            for stage, lout in stage_map.items():
                if lout is None:
                    continue
                d = dice_score(lout.logits, target)
                i = iou_score(lout.logits, target)
                entry = out.setdefault(lens_name, {}).setdefault(
                    stage, {"dice": [], "iou": []}
                )
                entry["dice"].append(d)
                entry["iou"].append(i)
    if uncertainty_per_image_stage is not None:
        for lens_name, stage_map in uncertainty_per_image_stage.items():
            for stage, std_list in stage_map.items():
                entry = out.setdefault(lens_name, {}).setdefault(
                    stage, {"dice": [], "iou": []}
                )
                entry["dice_uncertainty"] = list(std_list)
    return out


def save_depth_curve_csv(
    aggregated: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    path: str | Path,
) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_unc = any(
        "dice_uncertainty" in m for sm in aggregated.values() for m in sm.values()
    )
    fieldnames = ["lens", "stage", "n_images", "dice_mean", "dice_std", "iou_mean", "iou_std"]
    if has_unc:
        fieldnames += ["dice_uncertainty_mean", "dice_uncertainty_std"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for lens_name, stage_map in aggregated.items():
            for stage, metrics in stage_map.items():
                ds = metrics.get("dice", [])
                iss = metrics.get("iou", [])
                if not ds:
                    continue
                row = [
                    lens_name, stage, len(ds),
                    f"{sum(ds)/len(ds):.4f}",
                    f"{torch.tensor(ds).std().item():.4f}",
                    f"{sum(iss)/len(iss):.4f}",
                    f"{torch.tensor(iss).std().item():.4f}",
                ]
                if has_unc:
                    unc = metrics.get("dice_uncertainty", [])
                    if unc:
                        row += [
                            f"{sum(unc)/len(unc):.4f}",
                            f"{torch.tensor(unc).std().item():.4f}",
                        ]
                    else:
                        row += ["", ""]
                w.writerow(row)


def plot_depth_curves(
    aggregated: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    stages: Sequence[str] | None = None,
    metric: str = "dice",
    title: str = "Profundidad de decisión",
    save_path: str | Path | None = None,
    show_uncertainty_bands: bool = True,
):
    """Dibuja la curva ``metric-vs-stage`` para cada lente.

    Si hay campo ``dice_uncertainty`` en la agregación, dibuja bandas
    sombreadas ``±1 std`` alrededor de la curva.
    """
    import matplotlib.pyplot as plt

    if stages is None:
        s: set[str] = set()
        for stage_map in aggregated.values():
            s.update(stage_map.keys())
        stages = list(s)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"logit": "tab:gray", "tuned": "tab:orange", "jacobian": "tab:blue"}
    for lens_name, stage_map in aggregated.items():
        xs, ys, stds, ustd = [], [], [], []
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
            unc = ms.get("dice_uncertainty", [])
            ustd.append(sum(unc) / len(unc) if unc else 0.0)
        if xs:
            ax.errorbar(
                xs, ys, yerr=stds,
                label=lens_name, marker="o",
                color=colors.get(lens_name, None), capsize=2,
            )
            if show_uncertainty_bands and any(u > 0 for u in ustd):
                ax.fill_between(
                    xs,
                    [y - u for y, u in zip(ys, ustd)],
                    [y + u for y, u in zip(ys, ustd)],
                    color=colors.get(lens_name), alpha=0.15,
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
