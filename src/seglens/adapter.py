"""Adapter protocol y adaptadores concretos (UNet, UniVerSeg, genérico).

El Adapter es la única pieza modelo-específica de SegLens. Cualquier modelo que
exponga ``name``, ``stages``, ``segment()``, ``head()``, ``features()`` y
``preprocess()`` puede ser analizado por el resto del framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from seglens.hooks import FeatureExtractor


@runtime_checkable
class SegmentationAdapter(Protocol):
    """Contrato mínimo que cualquier modelo de segmentación debe cumplir.

    Atributos
    ---------
    name : str
        Nombre legible del modelo (p.ej. ``"unet"``, ``"universeg"``).
    stages : list[str]
        Nombres de las etapas introspectables. Cada ``features()`` devuelve
        un dict con estas claves y tensores ``(B, C_l, H_l, W_l)``.
    out_channels : int
        Número de canales de la salida de segmentación (1 para binaria).

    Métodos
    -------
    segment(x) -> Tensor
        Devuelve logits de segmentación ``(B, out_channels, H, W)``.
    head() -> nn.Module
        Devuelve la cabeza 1×1 final (el "unembedding" del modelo).
    features(x) -> dict[str, Tensor]
        Devuelve las features de cada etapa durante un forward. Si el modelo
        está en modo ``train``, se ejecuta en ``no_grad`` salvo que el
        llamador indique lo contrario.
    preprocess(x) -> Tensor
        Preprocesa la entrada al formato esperado por el modelo (resize,
        grayscale, normalización). El Adapter debe ser idempotente.
    """

    name: str
    stages: list[str]
    out_channels: int

    def segment(self, x: Tensor) -> Tensor: ...
    def head(self) -> nn.Module: ...
    def features(self, x: Tensor) -> dict[str, Tensor]: ...
    def preprocess(self, x: Tensor) -> Tensor: ...


class _BaseAdapter:
    """Funcionalidad común: gestión de hooks y validación de stage names."""

    def __init__(self, model: nn.Module, stages: list[str], name: str, out_channels: int = 1):
        self._model = model
        self.stages = list(stages)
        self.name = name
        self.out_channels = out_channels
        self._validate_stages()

    def _validate_stages(self) -> None:
        """Comprueba que cada nombre de stage existe como submódulo."""
        names = {n for n, _ in self._model.named_modules()}
        for s in self.stages:
            if s not in names:
                raise ValueError(
                    f"Stage '{s}' no existe en el modelo. "
                    f"Submódulos disponibles (primeros 20): "
                    f"{sorted(list(names))[:20]}"
                )

    @property
    def model(self) -> nn.Module:
        return self._model

    def segment(self, x: Tensor) -> Tensor:
        return self._model(x)

    def head(self) -> nn.Module:
        raise NotImplementedError

    def features(self, x: Tensor) -> dict[str, Tensor]:
        if not self.stages:
            return {}
        with FeatureExtractor(self._model, self.stages) as extractor:
            _ = self.segment(x)
        return extractor.features

    def preprocess(self, x: Tensor) -> Tensor:
        return x

    def eval(self) -> "_BaseAdapter":
        self._model.eval()
        return self

    def to(self, device: Any) -> "_BaseAdapter":
        self._model.to(device)
        return self


class UNetAdapter(_BaseAdapter):
    """Adapter para el UNet 2D del TFM (Conv-BN-ReLU, sin residual stream).

    Stages introspectadas: ``inc, down1..4, up1..4, outc`` (10 etapas).
    """

    DEFAULT_STAGES: list[str] = [
        "inc",
        "down1",
        "down2",
        "down3",
        "down4",
        "up1",
        "up2",
        "up3",
        "up4",
        "outc",
    ]

    def __init__(self, model: nn.Module, stages: list[str] | None = None):
        stages = stages or self.DEFAULT_STAGES
        super().__init__(model=model, stages=stages, name="unet", out_channels=1)

    def head(self) -> nn.Module:
        if not hasattr(self._model, "outc"):
            raise AttributeError("UNet debe tener atributo 'outc' (1x1 conv final)")
        return self._model.outc

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 64,
        device: str = "cpu",
    ) -> "UNetAdapter":
        """Carga el UNet desde un .pth compatible con el TFM.

        El state_dict se carga con ``strict=False`` para tolerar prefijos
        inesperados (``module.`` de DataParallel, etc.).
        """
        from seglens.third_party import UNet

        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
        )
        state = torch.load(str(checkpoint), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # quitar prefijo 'module.' si viene de DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            import warnings

            warnings.warn(f"UNetAdapter: claves faltantes al cargar: {missing[:5]}...")
        if unexpected:
            import warnings

            warnings.warn(f"UNetAdapter: claves inesperadas al cargar: {unexpected[:5]}...")
        model.to(device).eval()
        return cls(model)


class UniVerSegAdapter(_BaseAdapter):
    """Adapter para UniVerSeg (foundation model few-shot).

    UniVerSeg reescala las entradas a 128×128 y a grayscale. Su
    introspección depende de la versión de ``universeg`` instalada: se
    inspeccionan los módulos vía ``named_modules()`` y se filtran los
    nombres con palabras clave típicas (``down``, ``up``, ``cross``,
    ``final``). Si no se encuentra nada, se cae a un fallback de un solo
    stage (``output``).
    """

    _STAGE_KEYWORDS: tuple[str, ...] = (
        "down_",
        "up_",
        "cross",
        "final",
        "outc",
        "output",
        "seg_head",
    )

    def __init__(self, model: nn.Module, stages: list[str] | None = None, input_size: tuple[int, int] = (128, 128)):
        self.input_size = input_size
        if stages is None:
            stages = self._discover_stages(model)
        super().__init__(model=model, stages=stages, name="universeg", out_channels=1)
        self._context_images: Tensor | None = None
        self._context_masks: Tensor | None = None

    @staticmethod
    def _discover_stages(model: nn.Module) -> list[str]:
        """Encuentra stages introspectables por nombre en el modelo universeg.

        Estrategia: nombres únicos cuyo path completo contenga alguna
        palabra clave. Si nada coincide, devuelve ``[]`` como fallback
        (input-output only — no hay introspección por capas).
        """
        names = [n for n, _ in model.named_modules() if n]
        if not names:
            return []

        candidates: list[str] = []
        seen: set[str] = set()
        for kw in UniVerSegAdapter._STAGE_KEYWORDS:
            for n in names:
                if kw in n and n not in seen:
                    seen.add(n)
                    candidates.append(n)

        # Si tras el filtrado quedan demasiados (>25), podar a los más "profundos"
        # asumiendo que la numeración indica profundidad.
        if len(candidates) > 25:
            candidates = candidates[:25]

        return candidates

    def head(self) -> nn.Module:
        """Devuelve la cabeza final; busca módulos cuyo nombre contenga 'final' u 'output'."""
        for n, m in self._model.named_modules():
            if "final" in n or n.endswith("output") or n.endswith("seg_head"):
                if isinstance(m, nn.Conv2d):
                    return m
        # Fallback: la primera Conv2d de 1x1 del modelo
        for m in self._model.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                return m
        raise AttributeError("UniVerSegAdapter: no se encontró una cabeza 1x1 explícita")

    def set_context(self, images: Tensor, masks: Tensor) -> None:
        """Guarda el support set para few-shot.

        Acepta ``(N, 1, H, W)`` o ``(1, N, 1, H, W)`` para imágenes (1 canal)
        y máscaras. Internamente se almacena como ``(1, N, 1, H, W)``.

        Si la resolución no coincide con ``input_size``, se redimensiona.
        Para imágenes en color, reproyecta a 1 canal con promedio — esto
        es lo que hace UniVerSeg internamente.
        """
        import torch.nn.functional as F

        # Imágenes → (1, N, 1, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.dim() == 4 and images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        # Resize si hace falta
        if tuple(images.shape[-2:]) != self.input_size:
            imgs_flat = images.squeeze(0)  # (N, 1, H, W)
            imgs_flat = F.interpolate(imgs_flat, size=self.input_size, mode="bilinear", align_corners=False)
            images = imgs_flat.unsqueeze(0)

        # Máscaras → (1, N, 1, H, W)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if masks.dim() == 4 and masks.shape[0] != 1 and images.dim() == 5 and masks.shape[0] == images.shape[1]:
            # (N, 1, H, W) → (1, N, 1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # ya (1, ?, ?, ?) — debería ser (1, N, H, W), falta canal
            masks = masks.unsqueeze(2)  # (1, N, 1, H, W)
        if masks.dim() == 5 and tuple(masks.shape[-2:]) != self.input_size:
            masks_flat = masks.squeeze(0)  # (N, 1, H, W)
            masks_flat = F.interpolate(masks_flat, size=self.input_size, mode="nearest")
            masks = masks_flat.unsqueeze(0)

        # Validación final
        assert images.shape[0] == 1 and images.shape[2] == 1, \
            f"images shape {tuple(images.shape)} no es (1, N, 1, H, W)"
        assert masks.shape[0] == 1 and masks.shape[2] == 1, \
            f"masks shape {tuple(masks.shape)} no es (1, N, 1, H, W)"
        assert images.shape[1] == masks.shape[1], \
            f"N inconsistente: imgs={images.shape[1]} masks={masks.shape[1]}"

        # Mover al dispositivo del modelo si está disponible
        model_device = next(self._model.parameters(), torch.tensor(0)).device
        self._context_images = images.to(model_device)
        self._context_masks = masks.to(model_device)

    def segment(self, x: Tensor) -> Tensor:
        if self._context_images is None or self._context_masks is None:
            raise RuntimeError(
                "UniVerSegAdapter: debes llamar a set_context(images, masks) antes de segmentar"
            )
        x = self.preprocess(x)
        with torch.no_grad():
            return self._model(x, self._context_images, self._context_masks)

    def preprocess(self, x: Tensor) -> Tensor:
        """Resize a input_size y conversión a grayscale si la entrada es RGB."""
        import torch.nn.functional as F

        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    @classmethod
    def from_pretrained(
        cls, input_size: tuple[int, int] = (128, 128), device: str = "cpu"
    ) -> "UniVerSegAdapter":
        """Carga UniVerSeg pretrained (requiere ``pip install universeg``)."""
        try:
            from universeg import universeg
        except ImportError as e:
            raise ImportError(
                "UniVerSeg no está instalado. Instálalo con:\n"
                "  pip install git+https://github.com/JJGO/UniverSeg.git"
            ) from e
        model = universeg(pretrained=True).to(device).eval()
        return cls(model, input_size=input_size)


def make_adapter(model: nn.Module, name: str | None = None, **kwargs) -> SegmentationAdapter:
    """Factory: detecta automáticamente el Adapter a partir del modelo.

    Por convención:
    - atributo ``outc``  -> ``UNetAdapter``
    - callable con ``(image, support_imgs, support_masks)`` -> ``UniVerSegAdapter``
    - si nada encaja, error con mensaje claro.
    """
    if hasattr(model, "outc") and hasattr(model, "inc"):
        return UNetAdapter(model, **kwargs)

    import inspect

    try:
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.keys())
        if len(params) >= 3:
            return UniVerSegAdapter(model, **kwargs)
    except (TypeError, ValueError):
        pass

    raise ValueError(
        f"No se pudo inferir el Adapter a partir del modelo (name={name}). "
        f"Construye UNetAdapter o UniVerSegAdapter manualmente."
    )
