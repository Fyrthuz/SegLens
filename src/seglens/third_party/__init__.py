"""Re-implementación local del UNet 2D (equivalente a ``src.models.unet.UNet``
del TFM). Permite cargar checkpoints sin acoplar SegLens al TFM.

Re-exporta los símbolos desde :mod:`seglens.third_party.unet`.
"""

from seglens.third_party.unet import (  # noqa: F401
    UNet,
    DoubleConv,
    Down,
    Up,
    OutputConv,
    ConvBNReLU,
)

__all__ = ["UNet", "DoubleConv", "Down", "Up", "OutputConv", "ConvBNReLU"]
