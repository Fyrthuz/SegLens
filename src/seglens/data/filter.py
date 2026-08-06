"""Filtrado de slices LGG por porcentaje de foreground.

Mantiene los slices con al menos ``threshold``% de píxeles tumorales.
Equivalente a ``src.utils.filter_data_mri`` del TFM.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable


def _listdir(p: str) -> list[str]:
    try:
        return sorted(os.listdir(p))
    except FileNotFoundError:
        return []


def filter_dataset(
    src_root: str,
    dst_root: str,
    threshold: float = 0.01,
    verbose: bool = True,
) -> dict[str, int]:
    """Copia a ``dst_root`` los slices con foreground ratio ≥ ``threshold``.

    Returns
    -------
    ``{"total": int, "kept": int, "filtered_out": int}``
    """
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    from PIL import Image
    import numpy as np

    total = kept = 0
    for case in _listdir(src_root):
        case_path = os.path.join(src_root, case)
        if not os.path.isdir(case_path):
            continue
        dst_case = os.path.join(dst_root, case)
        os.makedirs(dst_case, exist_ok=True)
        for f in _listdir(case_path):
            if not (f.lower().endswith(".tif") and "_mask" in f):
                continue
            mask = np.array(Image.open(os.path.join(case_path, f)).convert("L"))
            ratio = (mask > 0).mean()
            total += 1
            if ratio >= threshold:
                kept += 1
                # copia imagen y máscara
                base, ext = os.path.splitext(f)
                img_file = base.replace("_mask", "") + ext
                shutil.copy2(os.path.join(case_path, img_file), os.path.join(dst_case, img_file))
                shutil.copy2(os.path.join(case_path, f), os.path.join(dst_case, f))
    out = {"total": total, "kept": kept, "filtered_out": total - kept}
    if verbose:
        print(
            f"[filter] {kept}/{total} slices kept "
            f"({kept / max(total, 1):.1%}), threshold={threshold}"
        )
    return out
