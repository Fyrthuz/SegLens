"""``seglens-download-data``: descarga el dataset LGG desde Kaggle y filtra.

Uso::

    uv run seglens-download-data --raw ./data/lgg_raw --filtered ./data/lgg --threshold 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and filter LGG MRI Segmentation dataset.")
    p.add_argument(
        "--raw",
        type=Path,
        default=Path("data/lgg_raw"),
        help="Directorio destino de la descarga cruda.",
    )
    p.add_argument(
        "--filtered",
        type=Path,
        default=Path("data/lgg"),
        help="Directorio destino del dataset filtrado por foreground ratio.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Mínimo ratio de píxeles foreground (0–1) para conservar un slice.",
    )
    p.add_argument(
        "--kaggle-username",
        type=str,
        default=os.environ.get("KAGGLE_USERNAME"),
        help="Usuario Kaggle (o env KAGGLE_USERNAME).",
    )
    p.add_argument(
        "--kaggle-key",
        type=str,
        default=os.environ.get("KAGGLE_KEY"),
        help="API key Kaggle (o env KAGGLE_KEY).",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Si se indica, asume que --raw ya contiene el dataset y solo filtra.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.raw
    filtered = args.filtered
    raw.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if not args.kaggle_username or not args.kaggle_key:
            print(
                "ERROR: se necesitan credenciales Kaggle. Define KAGGLE_USERNAME y "
                "KAGGLE_KEY o usa --skip-download si ya tienes el dataset.",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            import kaggle  # noqa: F401
        except ImportError:
            print("ERROR: pip install kaggle", file=sys.stderr)
            sys.exit(2)

        os.environ["KAGGLE_USERNAME"] = args.kaggle_username
        os.environ["KAGGLE_KEY"] = args.kaggle_key
        print(f"[download] Descargando LGG Segmentation a {raw}…")
        # Kaggle CLI: kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        zip_path = raw / "lgg-mri-segmentation.zip"
        api.dataset_download_files(
            "mateuszbuda/lgg-mri-segmentation",
            path=str(raw),
            unzip=False,
        )
        # unzip
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw)
        zip_path.unlink(missing_ok=True)

    # El dataset se extrae en un subdirectorio "lgg-mri-segmentation/" o similar.
    # Buscamos el directorio que contiene los TCGA_*
    candidates = [
        raw / "lgg-mri-segmentation" / "kaggle_3m",
        raw / "kaggle_3m",
        raw / "lgg-mri-segmentation",
        raw,
    ]
    src = next((c for c in candidates if c.exists() and any(p.is_dir() and p.name.startswith("TCGA") for p in c.iterdir())), None)
    if src is None:
        print(
            f"ERROR: no se encontró un subdirectorio con casos TCGA en {raw}. "
            f"Estructura: {[str(p) for p in raw.iterdir()]}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[filter] Filtrando {src} → {filtered} (threshold={args.threshold})…")
    from seglens.data.filter import filter_dataset

    filter_dataset(str(src), str(filtered), threshold=args.threshold)
    print(f"[filter] OK. Dataset en {filtered}.")


if __name__ == "__main__":
    main()
