"""Lentes causales (logit / tuned / jacobian) y orquestador de calibración."""

from seglens.lens.base import LensOutput
from seglens.lens.jacobian_lens import JacobianLens
from seglens.lens.logit_lens import LogitLens
from seglens.lens.tuned_lens import TunedLens

__all__ = [
    "LensOutput",
    "LogitLens",
    "TunedLens",
    "JacobianLens",
]


def calibrate_lenses(adapter, dataloader, n_images: int = 200, device: str = "cpu"):
    """Calibra las tres lentes sobre un dataloader de validación.

    Devuelve un dict ``{"logit": LogitLens|None, "tuned": TunedLens, "jacobian": JacobianLens}``.
    ``logit`` puede ser ``None`` si el Adapter reporta incompatibilidad de canales con la cabeza.
    """
    from seglens.lens.calibration import run_calibration

    return run_calibration(adapter, dataloader, n_images=n_images, device=device)
