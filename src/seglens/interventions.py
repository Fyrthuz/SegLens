"""Intervenciones sobre features de una etapa.

Análogas a las del paper original (ablación, steering, patching) pero
sobre mapas de features 4D en vez de residual streams.

Todas las funciones devuelven **nuevos tensores** (no mutan in-place).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def ablate_direction(features: Tensor, direction: Tensor, eps: float = 1e-8) -> Tensor:
    """Proyecta fuera la componente de ``features`` a lo largo de ``direction``.

    ``features`` tiene shape ``(B, C, H, W)`` y ``direction`` shape ``(C,)``.
    Devuelve ``f - <f, d̂> · d̂`` (en cada píxel).
    """
    d = direction.to(features.device).to(features.dtype)
    d = d / (d.norm() + eps)
    B, C, H, W = features.shape
    flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    coeff = flat @ d  # (B*H*W,)
    flat = flat - coeff.unsqueeze(-1) * d
    return flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def steer(features: Tensor, direction: Tensor, alpha: float = 1.0) -> Tensor:
    """Añade ``alpha · direction`` a cada píxel de ``features``."""
    d = direction.to(features.device).to(features.dtype)
    return features + alpha * d.view(1, -1, 1, 1)


def patch_activations(
    features: Tensor,
    source: Tensor,
    mask: Tensor,
) -> Tensor:
    """Copia activaciones de ``source`` en ``features`` donde ``mask`` es True.

    ``mask`` debe ser ``(B, 1, H, W)`` o ``(B, H, W)`` y se bilinearly-upsamplea
    si la resolución no coincide con ``features``.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != features.shape[-2:]:
        mask = F.interpolate(mask.float(), size=features.shape[-2:], mode="nearest")
    mask = (mask > 0.5).to(features.dtype)
    return features * (1 - mask) + source * mask


def ablate_concept(
    features: Tensor,
    cav: Tensor,
    strength: float = 1.0,
) -> Tensor:
    """Ablaciona la dirección del CAV con una fuerza ``strength ∈ [0, 1]``.

    ``strength=1`` elimina completamente la proyección; ``strength=0`` no hace nada.
    """
    d = cav.to(features.device).to(features.dtype)
    d = d / (d.norm() + 1e-8)
    B, C, H, W = features.shape
    flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    coeff = flat @ d
    flat = flat - strength * coeff.unsqueeze(-1) * d
    return flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
