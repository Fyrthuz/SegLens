"""Dataset LGG MRI Segmentation y utilidades de split por paciente.

Re-implementación local (basada en el TFM) para evitar acoplamiento.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def extract_patient_id(case_name: str) -> str:
    """Extrae el ID de paciente del nombre del directorio TCGA.

    Ejemplo: ``'TCGA_CS_4941_19960909'`` → ``'TCGA_CS_4941'``.
    """
    return "_".join(case_name.split("_")[:3])


def recover_image_mask_pairs(root_dir: str) -> Tuple[List[str], List[str]]:
    """Recorre ``root_dir`` y devuelve ``(image_paths, patient_ids)``."""
    pairs: List[str] = []
    patient_ids: List[str] = []
    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)
        if not os.path.isdir(case_path):
            continue
        patient_id = extract_patient_id(case)
        for file in sorted(os.listdir(case_path)):
            if file.lower().endswith(".tif") and "_mask" not in file:
                image_path = os.path.join(case_path, file)
                base, ext = os.path.splitext(file)
                mask_file = base + "_mask" + ext
                mask_path = os.path.join(case_path, mask_file)
                if os.path.exists(mask_path):
                    pairs.append(image_path)
                    patient_ids.append(patient_id)
    return pairs, patient_ids


class LGGSegmentationDataset(Dataset):
    """Dataset de segmentación LGG MRI.

    Parámetros
    ----------
    root_dir : str
        Directorio raíz con subdirectorios por paciente.
    image_transform, mask_transform : callable, opcional
        Transformaciones a aplicar a imagen / máscara. Por defecto, sólo
        ``ToTensor``.
    size : (int, int), opcional
        Si se indica, redimensiona imagen y máscara a este tamaño.
    """

    def __init__(
        self,
        root_dir: str,
        image_transform=None,
        mask_transform=None,
        size: tuple[int, int] | None = None,
    ):
        self.root_dir = root_dir
        self.image_paths, self.patient_ids = recover_image_mask_pairs(root_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.size = size
        if not self.image_paths:
            raise FileNotFoundError(
                f"No se encontraron imágenes en {root_dir}. ¿Has ejecutado download-data?"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        base, ext = os.path.splitext(image_path)
        mask_path = base + "_mask" + ext

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.size is not None:
            image = image.resize(self.size, Image.BILINEAR)
            mask = mask.resize(self.size, Image.NEAREST)

        if self.image_transform is None:
            image = transforms.ToTensor()(image)
        else:
            image = self.image_transform(image)

        if self.mask_transform is None:
            mask = transforms.ToTensor()(mask)
        else:
            mask = self.mask_transform(mask)

        # binarizar máscara (algunos .tif tienen valores continuos)
        mask = (mask > 0.5).float()
        return image, mask


def split_by_patient(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split por paciente para evitar data leakage.

    Devuelve ``(train_indices, val_indices, test_indices)``.
    """
    random.seed(seed)
    np.random.seed(seed)

    patient_to_indices: dict[str, list[int]] = {}
    for idx, pid in enumerate(patient_ids):
        patient_to_indices.setdefault(pid, []).append(idx)

    patients = list(patient_to_indices.keys())
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    train_indices = sorted([i for p in train_patients for i in patient_to_indices[p]])
    val_indices = sorted([i for p in val_patients for i in patient_to_indices[p]])
    test_indices = sorted([i for p in test_patients for i in patient_to_indices[p]])
    return train_indices, val_indices, test_indices


def save_test_indices(test_indices: List[int], path: str) -> None:
    with open(path, "w") as f:
        json.dump(test_indices, f)


def load_test_indices(path: str) -> List[int]:
    with open(path) as f:
        return json.load(f)
