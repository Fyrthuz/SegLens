"""E5: ablación de CAVs de conceptos por etapa.

Mide ΔDice al ablacionar la dirección de un concepto en una etapa dada,
comparado con una dirección aleatoria de la misma magnitud.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from seglens.adapter import SegmentationAdapter
from seglens.concepts import concept_scores, fit_cavs
from seglens.hooks import FeatureExtractor
from seglens.interventions import ablate_concept
from seglens.metrics import dice_score


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
):
    """Para cada ``(stage, concept)`` mide el ΔDice al ablacionar el CAV.

    Devuelve ``results[(stage, concept)] = {"clean_dice", "ablated_dice", "random_delta_dice"}``.
    """
    adapter.model.to(device).eval()
    results: dict[tuple[str, str], dict] = {}

    for stage, concept_cavs in cavs.items():
        for concept, cav in concept_cavs.items():
            clean_scores, ablated_scores, random_scores = [], [], []
            random_dirs = [random_direction_like(cav) for _ in range(n_random_controls)]
            random_results = [[] for _ in random_dirs]

            for batch in dataloader:
                x, y = batch[0], batch[1]
                x = adapter.preprocess(x).to(device)
                y = y.to(device).float()
                if y.dim() == 3:
                    y = y.unsqueeze(1)

                # 1) clean
                clean_y = adapter.model(x)
                for b in range(x.shape[0]):
                    clean_scores.append(dice_score(clean_y[b], y[b]))

                # 2) ablated: hook para modificar features en la etapa
                captured = {}

                def make_hook(name, direction, alpha):
                    def fn(_m, _i, out):
                        captured[name] = ablate_concept(out, direction, strength=alpha)

                    return fn

                handle = adapter.model.get_submodule(stage).register_forward_hook(
                    make_hook(stage, cav, strength)
                )
                try:
                    ablated_y = adapter.model(x)
                finally:
                    handle.remove()
                for b in range(x.shape[0]):
                    ablated_scores.append(dice_score(ablated_y[b], y[b]))

                # 3) random controls
                for r_idx, r_dir in enumerate(random_dirs):
                    captured_r = {}

                    def make_hook_r(name, direction):
                        def fn(_m, _i, out):
                            captured_r[name] = ablate_concept(out, direction, strength=strength)

                        return fn

                    handle_r = adapter.model.get_submodule(stage).register_forward_hook(
                        make_hook_r(stage, r_dir)
                    )
                    try:
                        rand_y = adapter.model(x)
                    finally:
                        handle_r.remove()
                    for b in range(x.shape[0]):
                        random_results[r_idx].append(dice_score(rand_y[b], y[b]))

            clean_mean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
            abl_mean = sum(ablated_scores) / len(ablated_scores) if ablated_scores else 0.0
            rand_means = [
                sum(rs) / len(rs) if rs else 0.0 for rs in random_results
            ]
            results[(stage, concept)] = {
                "clean_dice": clean_mean,
                "ablated_dice": abl_mean,
                "delta_dice": abl_mean - clean_mean,
                "random_delta_dice": [
                    rm - clean_mean for rm in rand_means
                ],
                "n_images": len(clean_scores),
            }
    return results
