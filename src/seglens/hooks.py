"""FeatureExtractor: context manager que captura features intermedias por nombre.

Análogo directo a "capturar la residual stream por capa" en el J-lens, pero
adaptado a modelos con stages de canales y resoluciones variables.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor


class FeatureExtractor(AbstractContextManager):
    """Registra forward hooks en módulos nombrados y captura sus salidas.

    Uso::

        with FeatureExtractor(model, ["inc", "down1", "up4", "outc"]) as ex:
            y = model(x)
        features = ex.features   # dict[str, Tensor]
    """

    def __init__(self, model: nn.Module, stages: Iterable[str], detach: bool = True):
        self._model = model
        self.stages = list(stages)
        self.detach = detach
        self.features: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._validate()

    def _validate(self) -> None:
        named = {n for n, _ in self._model.named_modules()}
        missing = [s for s in self.stages if s not in named]
        if missing:
            raise ValueError(
                f"FeatureExtractor: stages inexistentes: {missing}. "
                f"Disponibles (primeros 30): {sorted(named)[:30]}"
            )

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            t = out
            if self.detach:
                t = t.detach()
            # Si por algún motivo el mismo stage aparece más de una vez (no debería)
            if name in self.features:
                raise RuntimeError(f"Stage '{name}' capturado dos veces; revisa el grafo")
            self.features[name] = t

        return fn

    def __enter__(self) -> "FeatureExtractor":
        for stage in self.stages:
            mod = self._model.get_submodule(stage)
            self._handles.append(mod.register_forward_hook(self._hook(stage)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def resolve_spatial(features: Mapping[str, Tensor], target_hw: tuple[int, int]) -> dict[str, Tensor]:
    """Bilinearly upsamples cada feature a ``target_hw``. Devuelve un nuevo dict.

    Útil para que todas las etapas vivan en la misma cuadrícula antes de aplicar
    una lente o un probe.
    """
    import torch.nn.functional as F

    out: dict[str, Tensor] = {}
    for name, f in features.items():
        if f.dim() != 4:
            raise ValueError(f"Feature '{name}' no es 4D (B,C,H,W): shape={tuple(f.shape)}")
        if tuple(f.shape[-2:]) == target_hw:
            out[name] = f
        else:
            out[name] = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
    return out
