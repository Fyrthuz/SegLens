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


def calibrate_lenses(
    adapter, dataloader, n_images: int = 200, device: str = "cpu",
    max_pixels_per_image: int | None = 64,
):
    """Calibra las tres lentes sobre un dataloader de validación.

    Parámetros
    ----------
    max_pixels_per_image : int | None
        Sub-muestreo espacial para el J-lens. Si ``None``, se usan todos los
        píxeles (más preciso pero más lento). Para imágenes grandes en GPU,
        valores entre 16 y 64 evitan OOM manteniendo suficiente estadística.
    """
    from seglens.lens.calibration import run_calibration

    return run_calibration(
        adapter, dataloader, n_images=n_images, device=device,
        max_pixels_per_image=max_pixels_per_image,
    )
