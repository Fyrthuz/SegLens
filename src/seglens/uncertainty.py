"""Cuantificación de incertidumbre agnóstica del modelo.

Tres métodos soportados:

- **MC Dropout** (Gal & Ghahramani 2015): N forward passes con dropout
  estocástico activo en las features de ciertos stages. Implementado vía
  hooks (``MCDropoutInjector``) para no requerir reentrenamiento.
- **TTA**: N forward passes con transforms geométricas/intensidad. Las
  predicciones se promedian y se devuelve su varianza.
- **Noisy**: N forward passes con ruido gaussiano en la entrada.

El ``UncertaintyEstimator`` es agnóstico del modelo: opera sobre cualquier
``SegmentationAdapter`` y devuelve ``mean``, ``std`` y ``entropy`` (per-pixel).
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Hook-based MC Dropout
# --------------------------------------------------------------------------- #


class MCDropoutInjector(AbstractContextManager):
    """Context manager: añade dropout estocástico a features de un stage.

    A diferencia de ``model.train()``, este injector solo activa dropout en
    los stages especificados — el resto del modelo queda en ``eval()``. Esto
    evita afectar a BatchNorm running stats.

    Importante: las features dropeadas tienen un gradiente distinto al del
    forward limpio. No usar durante entrenamiento.
    """

    def __init__(
        self,
        model: nn.Module,
        stages: Sequence[str],
        p: float = 0.1,
        mode: str = "channel",  # "channel" | "spatial" | "element"
    ):
        if not 0.0 < p < 1.0:
            raise ValueError(f"p debe estar en (0, 1); recibido {p}")
        if mode not in ("channel", "spatial", "element"):
            raise ValueError(f"mode debe ser uno de channel/spatial/element; recibido {mode}")
        self.model = model
        self.stages = list(stages)
        self.p = p
        self.mode = mode
        self._handles: list = []
        self._validate()

    def _validate(self) -> None:
        names = {n for n, _ in self.model.named_modules()}
        missing = [s for s in self.stages if s not in names]
        if missing:
            raise ValueError(
                f"MCDropoutInjector: stages inexistentes: {missing}. "
                f"Disponibles (primeros 30): {sorted(names)[:30]}"
            )

    def _make_hook(self, p: float, mode: str):
        def fn(_m, _i, out):
            if isinstance(out, tuple):
                # Capturamos solo la primera rama (target) — análoga a
                # como lo hace universeg_feature_extractor.
                out = out[0]
            return F.dropout(out, p=p, training=True)
        return fn

    def __enter__(self) -> "MCDropoutInjector":
        for stage in self.stages:
            mod = self.model.get_submodule(stage)
            self._handles.append(
                mod.register_forward_hook(self._make_hook(self.p, self.mode))
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# TTA transforms
# --------------------------------------------------------------------------- #


def _tta_transforms_unet(with_rotation: bool = True) -> list[tuple[str, callable, callable]]:
    """Transforms para UNet (256×256, sin restricción de tamaño).

    IMPORTANTE: ``torch.rot90`` en PyTorch ≥ 2.x usa ``dims=[0, 1]`` por defecto
    (rota batch y canales), NO ``[-2, -1]``. Hay que especificar dims explícitamente.
    """
    out: list[tuple[str, callable, callable]] = [
        ("identity", lambda x: x, lambda y: y),
        ("hflip", lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
        ("vflip", lambda x: torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),
    ]
    if with_rotation:
        out.extend([
            ("rot90",
             lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=-1, dims=[-2, -1])),
            ("rot180",
             lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=2, dims=[-2, -1])),
            ("rot270",
             lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
             lambda y: torch.rot90(y, k=-3, dims=[-2, -1])),
        ])
    return out


def _tta_transforms_universeg(with_rotation: bool = True) -> list[tuple[str, callable, callable]]:
    """Transforms para UniVerSeg (resize interno 128×128, solo hflip."""
    return [
        ("identity", lambda x: x, lambda y: y),
        ("hflip", lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
    ]


# --------------------------------------------------------------------------- #
# Default stages
# --------------------------------------------------------------------------- #


def _default_mc_stages(adapter) -> list[str]:
    """Heurística de stages donde el MC Dropout es más informativo.

    Por defecto, las dos últimas etapas del decoder — que es donde se
    forma la decisión según los runs previos.
    """
    name = getattr(adapter, "name", "unet")
    if name == "unet":
        return ["up3", "up4"]
    if name == "universeg":
        return ["dec_blocks.2.target.vmapped", "dec_blocks.1.target.vmapped"]
    return adapter.stages[-2:] if adapter.stages else []


# --------------------------------------------------------------------------- #
# UncertaintyEstimator
# --------------------------------------------------------------------------- #


class UncertaintyEstimator:
    """Agnóstico del modelo. Métodos: mc_dropout, tta, noisy."""

    def __init__(
        self,
        adapter,
        mc_stages: Sequence[str] | None = None,
        mc_p: float = 0.1,
        n_samples: int = 10,
        tta_transforms: list | None = None,
        tta_with_rotation: bool = True,
        noise_sigma: float = 0.01,
        device: str | None = None,
    ):
        self.adapter = adapter
        self.mc_stages = list(mc_stages) if mc_stages else _default_mc_stages(adapter)
        self.mc_p = mc_p
        self.n_samples = int(n_samples)
        self.noise_sigma = float(noise_sigma)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if tta_transforms is not None:
            self.tta_transforms = tta_transforms
        elif adapter.name == "unet":
            self.tta_transforms = _tta_transforms_unet(with_rotation=tta_with_rotation)
        elif adapter.name == "universeg":
            self.tta_transforms = _tta_transforms_universeg(with_rotation=tta_with_rotation)
        else:
            self.tta_transforms = [("identity", lambda x: x, lambda y: y)]

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_one(self, x: Tensor) -> Tensor:
        return self.adapter.segment(x)

    # ------------------------------------------------------------------ #
    # MC Dropout
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def mc_dropout(self, x: Tensor) -> dict[str, Tensor]:
        """N forward passes con dropout estocástico en ``mc_stages``."""
        x = self.adapter.preprocess(x).to(self.device)
        if not x.dim() == 4:
            raise ValueError(f"x debe ser 4D (B,C,H,W); shape={tuple(x.shape)}")
        predictions: list[Tensor] = []
        with MCDropoutInjector(self.adapter.model, self.mc_stages, p=self.mc_p):
            for _ in range(self.n_samples):
                y = self._forward_one(x)
                predictions.append(y.detach())
        stacked = torch.stack(predictions, dim=0)  # (N, B, K, H, W)
        return self._aggregate(stacked)

    # ------------------------------------------------------------------ #
    # TTA
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def tta(self, x: Tensor) -> dict[str, Tensor]:
        """Aplica cada transform N veces y agrega."""
        x = self.adapter.preprocess(x).to(self.device)
        predictions: list[Tensor] = []
        for _, x_fn, y_inv in self.tta_transforms:
            x_t = x_fn(x)
            for _ in range(self.n_samples):
                y_t = self._forward_one(x_t)
                y = y_inv(y_t)
                predictions.append(y.detach())
        stacked = torch.stack(predictions, dim=0)  # (T*N, B, K, H, W)
        return self._aggregate(stacked)

    # ------------------------------------------------------------------ #
    # Noisy (fallback)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def noisy(self, x: Tensor) -> dict[str, Tensor]:
        """N forward passes con ruido gaussiano en la entrada."""
        x = self.adapter.preprocess(x).to(self.device)
        predictions: list[Tensor] = []
        for _ in range(self.n_samples):
            x_n = x + self.noise_sigma * torch.randn_like(x)
            y = self._forward_one(x_n)
            predictions.append(y.detach())
        stacked = torch.stack(predictions, dim=0)
        return self._aggregate(stacked)

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate(stacked: Tensor) -> dict[str, Tensor]:
        """stacked: (N, B, K, H, W) → mean/std/entropy per-pixel."""
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False)
        std = var.sqrt()
        # Entropy sobre la media (probabilidad promedio)
        p = torch.sigmoid(mean)
        eps = 1e-8
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        return {"mean": mean, "std": std, "var": var, "entropy": entropy,
                "predictions": stacked}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def uncertainty_map(self, x: Tensor, method: str = "mc_dropout") -> dict[str, Tensor]:
        method = method.lower()
        if method == "mc_dropout":
            return self.mc_dropout(x)
        if method == "tta":
            return self.tta(x)
        if method == "noisy":
            return self.noisy(x)
        raise ValueError(f"method debe ser mc_dropout/tta/noisy; recibido {method}")

    @torch.no_grad()
    def calibrate_baseline(self, dataloader, n: int = 20) -> dict[str, float]:
        """Calcula métricas de referencia de incertidumbre sobre n imágenes.

        Útil para saber si la incertidumbre es 'normal' antes de comparar
        con imágenes anómalas.
        """
        entropies: list[float] = []
        seen = 0
        for batch in dataloader:
            if seen >= n:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            out = self.uncertainty_map(x, method="mc_dropout")
            entropies.append(out["entropy"].mean().item())
            seen += x.shape[0]
        if not entropies:
            return {"mean_entropy": 0.0, "std_entropy": 0.0, "n_images": 0}
        return {
            "mean_entropy": float(sum(entropies) / len(entropies)),
            "std_entropy": float(torch.tensor(entropies).std().item()),
            "n_images": len(entropies),
        }
