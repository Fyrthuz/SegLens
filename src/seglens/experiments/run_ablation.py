"""E5: ablación de CAVs de conceptos por etapa.

Mide ΔDice y ΔUncertainty al ablacionar la dirección de un concepto en una
etapa dada, comparado con una dirección aleatoria de la misma magnitud.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from seglens.adapter import SegmentationAdapter
from seglens.concepts import concept_scores, fit_cavs
from seglens.hooks import FeatureExtractor
from seglens.interventions import ablate_concept
from seglens.metrics import dice_score


def _entropy_from_logits(logits: Tensor) -> float:
    """Entropy binaria promedio del mapa de logits."""
    p = torch.sigmoid(logits)
    eps = 1e-8
    h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return float(h.mean().item())


def random_direction_like(v: Tensor) -> Tensor:
    """Vector aleatorio de la esfera unidad con la misma magnitud que ``v``."""
    r = torch.randn_like(v)
    return r / (r.norm() + 1e-8)


@torch.no_grad()
def run_ablation(
    adapter: SegmentationAdapter,
    cavs: Mapping[str, Mapping[str, Tensor]],
    dataloader: DataLoader,
    device: str = "cpu",
    strength: float = 1.0,
    n_random_controls: int = 5,
    n_uncertainty_samples: int = 0,
):
    """Para cada ``(stage, concept)`` mide ΔDice y opcionalmente ΔEntropy.

    Parámetros
    ----------
    n_uncertainty_samples : int
        Si > 0, calcula además ``clean_entropy``, ``ablated_entropy`` y
        ``delta_entropy`` usando N forward passes con MC Dropout / TTA
        / Noisy (default: ``adapter.uncertainty`` con método ``"mc_dropout"``).
    """
    adapter.model.to(device).eval()
    results: dict[tuple[str, str], dict] = {}

    for stage, concept_cavs in cavs.items():
        for concept, cav in concept_cavs.items():
            clean_scores, ablated_scores, random_scores = [], [], []
            clean_unc, ablated_unc, random_unc = [], [], []
            random_dirs = [random_direction_like(cav) for _ in range(n_random_controls)]
            random_results = [[] for _ in random_dirs]
            random_unc_results = [[] for _ in random_dirs]

            for batch in dataloader:
                x, y = batch[0], batch[1]
                x = adapter.preprocess(x).to(device)
                y = y.to(device).float()
                if y.dim() == 3:
                    y = y.unsqueeze(1)

                # 1) clean
                clean_y = adapter.segment(x)
                for b in range(x.shape[0]):
                    clean_scores.append(dice_score(clean_y[b], y[b]))
                if n_uncertainty_samples > 0:
                    u = adapter.uncertainty(x, method="mc_dropout",
                                             n_samples=n_uncertainty_samples)
                    clean_unc.append(_entropy_from_logits(u["mean"]))

                # 2) ablated
                def make_hook(direction, alpha):
                    def fn(_m, _i, out):
                        return ablate_concept(out, direction, strength=alpha)
                    return fn

                handle = adapter.model.get_submodule(stage).register_forward_hook(
                    make_hook(cav, strength)
                )
                try:
                    ablated_y = adapter.segment(x)
                finally:
                    handle.remove()
                for b in range(x.shape[0]):
                    ablated_scores.append(dice_score(ablated_y[b], y[b]))
                if n_uncertainty_samples > 0:
                    u = adapter.uncertainty(x, method="mc_dropout",
                                             n_samples=n_uncertainty_samples)
                    ablated_unc.append(_entropy_from_logits(u["mean"]))

                # 3) random controls
                for r_idx, r_dir in enumerate(random_dirs):
                    handle_r = adapter.model.get_submodule(stage).register_forward_hook(
                        make_hook(r_dir, strength)
                    )
                    try:
                        rand_y = adapter.segment(x)
                    finally:
                        handle_r.remove()
                    for b in range(x.shape[0]):
                        random_results[r_idx].append(dice_score(rand_y[b], y[b]))
                    if n_uncertainty_samples > 0:
                        u = adapter.uncertainty(x, method="mc_dropout",
                                                 n_samples=n_uncertainty_samples)
                        random_unc_results[r_idx].append(_entropy_from_logits(u["mean"]))

            n = len(clean_scores)
            clean_mean = sum(clean_scores) / n if n else 0.0
            abl_mean = sum(ablated_scores) / n if n else 0.0
            rand_means = [sum(rs) / n if n else 0.0 for rs in random_results]
            entry = {
                "clean_dice": clean_mean,
                "ablated_dice": abl_mean,
                "delta_dice": abl_mean - clean_mean,
                "random_delta_dice": [rm - clean_mean for rm in rand_means],
                "n_images": n,
            }
            if n_uncertainty_samples > 0 and clean_unc:
                cu = sum(clean_unc) / len(clean_unc)
                au = sum(ablated_unc) / len(ablated_unc)
                ru = [sum(rr) / len(rr) if rr else 0.0 for rr in random_unc_results]
                entry["clean_entropy"] = cu
                entry["ablated_entropy"] = au
                entry["delta_entropy"] = au - cu
                entry["random_delta_entropy"] = [r - cu for r in ru]

            results[(stage, concept)] = entry
    return results
