"""Tipos comunes a las tres lentes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class LensOutput:
    """Resultado de aplicar una lente a las features de una imagen.

    Attributes
    ----------
    logits : Tensor
        Mapa de logits ``(B, out_channels, H, W)`` a la resolución de la salida
        del modelo. Si la lente requiere upsampling (resolución de la feature
        < resolución de salida), se bilinearly-upsamplea la predicción.
    stage : str
        Nombre de la etapa leída.
    lens : str
        Identificador de la lente (``"logit"``, ``"tuned"``, ``"jacobian"``).
    extra : dict
        Metadatos libres: normas, magnitudes, scores intermedios, etc.
    """

    logits: Tensor
    stage: str
    lens: str
    extra: dict = field(default_factory=dict)

    def to(self, device) -> "LensOutput":
        return LensOutput(
            logits=self.logits.to(device),
            stage=self.stage,
            lens=self.lens,
            extra={k: v.to(device) if isinstance(v, Tensor) else v for k, v in self.extra.items()},
        )


def expected_input_channels(head: torch.nn.Module) -> int:
    """Infiere el nº de canales de entrada de la cabeza 1x1."""
    if hasattr(head, "conv") and isinstance(head.conv, torch.nn.Conv2d):
        return head.conv.in_channels
    if isinstance(head, torch.nn.Conv2d):
        return head.in_channels
    raise AttributeError("No se pudo inferir in_channels de la cabeza")
