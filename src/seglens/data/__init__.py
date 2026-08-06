"""Utilidades de datos: loaders para segmentación médica.

Actualmente soporta el dataset LGG MRI Segmentation (Kaggle). El formato
esperado es un árbol de directorios con un subdirectorio por paciente que
contiene imágenes ``*_1.tif`` y máscaras ``*_1_mask.tif`` (TCGA-style).
"""

from seglens.data.dataset import (
    LGGSegmentationDataset,
    extract_patient_id,
    load_test_indices,
    recover_image_mask_pairs,
    save_test_indices,
    split_by_patient,
)

__all__ = [
    "LGGSegmentationDataset",
    "extract_patient_id",
    "recover_image_mask_pairs",
    "split_by_patient",
    "save_test_indices",
    "load_test_indices",
]
