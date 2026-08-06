"""SegLens: model-agnostic causal lenses for medical image segmentation explainability."""

from seglens.adapter import SegmentationAdapter, UNetAdapter
from seglens.hooks import FeatureExtractor
from seglens.lens import (
    JacobianLens,
    LensOutput,
    LogitLens,
    TunedLens,
    calibrate_lenses,
)

__version__ = "0.1.0"

__all__ = [
    "SegmentationAdapter",
    "UNetAdapter",
    "FeatureExtractor",
    "LogitLens",
    "TunedLens",
    "JacobianLens",
    "LensOutput",
    "calibrate_lenses",
]
