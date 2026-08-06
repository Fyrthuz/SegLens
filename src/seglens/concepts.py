"""Conceptos clínicos derivados de T1c (LGG / BraTS-light).

Análogo del "vocabulario de tokens" del J-lens, pero los conceptos son
direcciones en espacio de features definidas por conjuntos de píxeles
humanamente interpretables (estilo TCAV, Kim et al. 2018).

Conceptos por defecto (sobre T1c, canal G en RGB):

- ``tumor_core``: Foreground ∩ T1c > p90(FG)   → alta intensidad tumoral
- ``tumor_periphery``: Foreground ∩ p60 ≤ T1c ≤ p90 (FG) → anillo perilesional
- ``ventricle_csf``: Background ∩ T1c < p15 (BG)  → LCR / ventrículos
- ``normal_white_matter``: Background ∩ p40 ≤ T1c ≤ p70 (BG)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CONCEPTS: dict[str, tuple[str, tuple[float, float] | None]] = {
    "tumor_core": ("fg", (0.90, 1.00)),
    "tumor_periphery": ("fg", (0.60, 0.90)),
    "ventricle_csf": ("bg", (0.00, 0.15)),
    "normal_white_matter": ("bg", (0.40, 0.70)),
}


@dataclass
class ConceptThresholds:
    """Umbrales por concepto en percentiles de T1c dentro de FG o BG."""

    fg_percentiles: Tensor  # shape (101,) — p_k de T1c sobre foreground
    bg_percentiles: Tensor  # shape (101,) — p_k de T1c sobre background

    @classmethod
    def from_batch(
        cls,
        images: Tensor,
        masks: Tensor,
        t1c_channel: int = 1,
    ) -> "ConceptThresholds":
        """Calcula percentiles 0–100 sobre las imágenes del batch de calibración.

        Parameters
        ----------
        images : (B, C, H, W)
            Imágenes RGB; el canal T1c se selecciona por ``t1c_channel``.
        masks : (B, 1, H, W) o (B, H, W)
            Máscaras binarias (foreground = tumor).
        """
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        t1c = images[:, t1c_channel : t1c_channel + 1]  # (B, 1, H, W)
        fg = t1c[masks > 0.5].flatten()
        bg = t1c[masks <= 0.5].flatten()
        if fg.numel() == 0 or bg.numel() == 0:
            raise ValueError("El batch no contiene píxeles FG o BG")
        ps = torch.linspace(0, 1, 101, device=images.device)
        fg_p = torch.quantile(fg.float(), ps)
        bg_p = torch.quantile(bg.float(), ps)
        return cls(fg_percentiles=fg_p, bg_percentiles=bg_p)

    def value_at(self, which: str, q: float) -> float:
        arr = self.fg_percentiles if which == "fg" else self.bg_percentiles
        idx = int(round(q * 100))
        idx = max(0, min(100, idx))
        return float(arr[idx].item())


def derive_concept_masks(
    images: Tensor,
    masks: Tensor,
    thresholds: ConceptThresholds,
    t1c_channel: int = 1,
    concepts: Mapping[str, tuple[str, tuple[float, float]]] | None = None,
) -> dict[str, Tensor]:
    """Devuelve ``{nombre_concepto: mask_bool}`` para cada imagen del batch.

    Cada mask tiene shape ``(B, H, W)`` y es True donde el píxel cumple la
    regla de percentil para el concepto.
    """
    concepts = concepts or DEFAULT_CONCEPTS
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    t1c = images[:, t1c_channel : t1c_channel + 1]  # (B, 1, H, W)
    fg_mask = masks > 0.5
    out: dict[str, Tensor] = {}
    for name, (which, (lo, hi)) in concepts.items():
        base = fg_mask if which == "fg" else ~fg_mask
        lo_val = thresholds.value_at(which, lo)
        hi_val = thresholds.value_at(which, hi)
        intensity_in_range = (t1c >= lo_val) & (t1c <= hi_val)
        out[name] = (base & intensity_in_range).squeeze(1)  # (B, H, W)
    return out


def fit_cavs(
    features_per_image: dict[str, Tensor],
    concept_masks_per_image: dict[str, Tensor],
    stage: str,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Ajusta un CAV (concept activation vector) por concepto en una etapa.

    Parameters
    ----------
    features_per_image : ``{stage: (B, C_l, H_l, W_l)}``
    concept_masks_per_image : ``{concept: (B, H_l, W_l)}`` (o compatible)
        Si las resoluciones no coinciden, las máscaras se bilinearly-upsamplean
        a la cuadrícula de las features.

    Returns
    -------
    ``{concept: cav}`` con cav de shape ``(C_l,)``.
    """
    f = features_per_image[stage]  # (B, C, H, W)
    B, C, H, W = f.shape
    cavs: dict[str, Tensor] = {}
    flat_f = f.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    for concept, masks in concept_masks_per_image.items():
        m = masks
        if m.dim() == 4:
            m = m.squeeze(1)
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
        flat_m = m.reshape(-1).bool()
        if flat_m.sum() < 2:
            cavs[concept] = torch.zeros(C, device=f.device, dtype=f.dtype)
            continue
        pos = flat_f[flat_m].mean(dim=0)
        neg = flat_f[~flat_m].mean(dim=0)
        cav = pos - neg
        cav = cav / (cav.norm() + eps)
        cavs[concept] = cav
    return cavs


def concept_scores(
    features: Tensor,  # (B, C, H, W)
    cavs: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Proyecta cada píxel sobre cada CAV: ``score[b,c,h,w] = <f, v_c>``.

    Devuelve ``{concept: (B, 1, H, W)}``.
    """
    B, C, H, W = features.shape
    flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    out: dict[str, Tensor] = {}
    for name, v in cavs.items():
        v = v.to(features.device).to(features.dtype)
        scores = flat @ v  # (B*H*W,)
        scores = scores.reshape(B, H, W).unsqueeze(1)  # (B, 1, H, W)
        out[name] = scores
    return out
