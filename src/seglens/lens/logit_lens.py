"""Logit lens: aplica la cabeza 1x1 directamente a las features de una etapa.

Análogo del "logit lens" del paper de Anthropic. En transformers funciona
porque la residual stream tiene ancho constante y la cabeza espera ese
ancho; en segmentación **solo es válido en la última etapa** (mismo nº de
canales que la cabeza). En etapas tempranas la cabeza es incompatible y la
lente devuelve ``None`` con un aviso.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from seglens.lens.base import LensOutput, expected_input_channels


class LogitLens:
    """Lente 1: aplica la cabeza final a features de cualquier etapa.

    Solo produce salida válida si ``feature_channels == head.in_channels``.
    En otro caso devuelve ``None`` y se sugiere usar :class:`TunedLens` o
    :class:`JacobianLens`.
    """

    name = "logit"

    def __init__(self, head: torch.nn.Module, target_hw: tuple[int, int]):
        self.head = head
        self.target_hw = target_hw
        self._expected = expected_input_channels(head)

    def compatible(self, features: Tensor) -> bool:
        return features.shape[1] == self._expected

    def __call__(self, stage: str, features: Tensor) -> Optional[LensOutput]:
        if not self.compatible(features):
            warnings.warn(
                f"[LogitLens] stage='{stage}' channels={features.shape[1]} "
                f"!= head.in_channels={self._expected}; saltando. Usa TunedLens.",
                stacklevel=2,
            )
            return None

        # Upsample a la resolución de la cabeza si hace falta
        if tuple(features.shape[-2:]) != self.target_hw:
            features = F.interpolate(
                features, size=self.target_hw, mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            logits = self.head(features)
        return LensOutput(logits=logits, stage=stage, lens=self.name, extra={})

    def per_image(self, stage: str, features_bchw: Tensor) -> Optional[Tensor]:
        """Devuelve solo el tensor de logits (sin metadatos) para una imagen."""
        out = self(stage, features_bchw)
        return out.logits if out is not None else None
