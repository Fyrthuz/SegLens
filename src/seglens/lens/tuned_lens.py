"""Tuned lens: probe lineal por etapa, entrenada por mínimos cuadrados.

Análogo del "tuned lens" del paper de Anthropic. En transformers se entrena
un mapa lineal por capa que aproxima la salida del modelo desde esa capa;
aquí entrenamos ``W_l ∈ R^{C_out × C_l}`` con ``out_channels`` filas (una
por canal de salida) y ``C_l`` columnas (canales de la feature de la
etapa). Se entrena por mínimos cuadrados sobre un split de calibración.

Es **correlacional** (no causal): aprende la mejor lectura lineal de las
features, pero no modela efectos de propagación.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from seglens.lens.base import LensOutput


@dataclass
class _Probe:
    weight: Tensor  # (C_out, C_l)
    bias: Tensor | None  # (C_out,)


class TunedLens:
    """Lente 2: probe lineal por etapa, entrenada por mínimos cuadrados.

    Uso::

        lens = TunedLens(target_hw=(H, W), out_channels=1)
        lens.fit(stage_features={"inc": ..., "up4": ...}, target_logits=y)
        out = lens("up4", features_up4)
    """

    name = "tuned"

    def __init__(self, target_hw: tuple[int, int], out_channels: int = 1):
        self.target_hw = target_hw
        self.out_channels = out_channels
        self._probes: dict[str, _Probe] = {}

    @torch.no_grad()
    def fit(
        self,
        stage_features: Mapping[str, Tensor],
        target_logits: Tensor,
        l2: float = 1e-4,
    ) -> "TunedLens":
        """Ajusta una probe por etapa con mínimos cuadrados regularizados.

        Parameters
        ----------
        stage_features : dict[str, Tensor]
            ``(N_total_pixels, C_l)`` apilado por etapa.
        target_logits : Tensor
            ``(N_total_pixels, out_channels)`` logit target.
        """
        for stage, feats in stage_features.items():
            if feats.dim() != 2:
                raise ValueError(
                    f"stage_features['{stage}'] debe ser 2D (Npixels, C_l); "
                    f"recibido shape={tuple(feats.shape)}"
                )
            if feats.shape[0] != target_logits.shape[0]:
                raise ValueError(
                    f"Nº de píxeles inconsistente en stage '{stage}': "
                    f"{feats.shape[0]} vs target {target_logits.shape[0]}"
                )
            # Resolver mínimos cuadrados con regularización ridge
            N, C = feats.shape
            K = target_logits.shape[1]
            X = feats  # (N, C)
            Y = target_logits  # (N, K)
            XtX = X.T @ X + l2 * torch.eye(C, device=X.device, dtype=X.dtype)
            XtY = X.T @ Y
            try:
                W = torch.linalg.solve(XtX, XtY)  # (C, K)
            except RuntimeError:
                # fallback: lstsq
                W = torch.linalg.lstsq(XtX, XtY).solution  # (C, K)
            self._probes[stage] = _Probe(weight=W.T.contiguous(), bias=None)  # (K, C)
        return self

    def has(self, stage: str) -> bool:
        return stage in self._probes

    @property
    def stages(self) -> list[str]:
        return list(self._probes.keys())

    @torch.no_grad()
    def __call__(self, stage: str, features: Tensor) -> LensOutput:
        if stage not in self._probes:
            raise KeyError(f"TunedLens sin probe entrenada para stage '{stage}'")
        probe = self._probes[stage]
        # Upsample a target_hw
        if tuple(features.shape[-2:]) != self.target_hw:
            features = F.interpolate(
                features, size=self.target_hw, mode="bilinear", align_corners=False
            )
        B, C, H, W = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        out = flat @ probe.weight.T  # (B*H*W, K)
        if probe.bias is not None:
            out = out + probe.bias
        K = out.shape[-1]
        logits = out.reshape(B, H, W, K).permute(0, 3, 1, 2).contiguous()  # (B,K,H,W)
        return LensOutput(logits=logits, stage=stage, lens=self.name, extra={})
