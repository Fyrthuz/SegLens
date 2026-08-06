"""``seglens-depth-curves``: calibra las 3 lentes y mide Dice/IoU por etapa."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UNetAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.lens import calibrate_lenses
from seglens.experiments.run_depth_curves import run_depth_curves, summarize_depth_curves
from seglens.viz import plot_depth_curves, save_depth_curve_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pth del UNet.")
    p.add_argument("--output-dir", type=Path, default=Path("results/depth_curves"))
    p.add_argument("--n-calibration", type=int, default=50)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-uncertainty", action="store_true",
                   help="Añade bandas de incertidumbre (MC Dropout) en la figura.")
    p.add_argument("--n-uncertainty-samples", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Dataset y split
    base_ds = LGGSegmentationDataset(str(args.data_root), size=(args.size, args.size))
    train_idx, val_idx, test_idx = split_by_patient(
        base_ds.patient_ids, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )
    calib_idx = train_idx[: args.n_calibration]
    if args.n_test < len(test_idx):
        test_idx = test_idx[: args.n_test]
    print(f"[depth-curves] calib={len(calib_idx)}  test={len(test_idx)}")

    # Adapter
    adapter = UNetAdapter.from_checkpoint(
        str(args.checkpoint), in_channels=3, out_channels=1, device=args.device
    )

    # Calibración
    calib_loader = DataLoader(Subset(base_ds, calib_idx), batch_size=args.batch_size)
    print("[depth-curves] Calibrando lentes…")
    # El J-lens itera por píxel — sub-muestreamos para evitar OOM en GPU
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    lenses = calibrate_lenses(
        adapter, calib_loader, n_images=args.n_calibration, device=args.device,
        max_pixels_per_image=32,
    )

    # Curva de profundidad
    test_loader = DataLoader(Subset(base_ds, test_idx), batch_size=args.batch_size)
    print("[depth-curves] Ejecutando en test…")
    per_image, targets = run_depth_curves(
        adapter, lenses, test_loader, max_images=args.n_test, device=args.device,
    )
    agg = summarize_depth_curves(per_image, targets)

    # Incertidumbre opcional: por (lente, stage, imagen), std de N MC passes
    if args.with_uncertainty:
        print(f"[depth-curves] Computando bandas de incertidumbre (N={args.n_uncertainty_samples})…")
        from seglens.uncertainty import UncertaintyEstimator
        from seglens.hooks import FeatureExtractor
        from seglens.metrics import dice_score
        import numpy as np

        est = UncertaintyEstimator(
            adapter, n_samples=args.n_uncertainty_samples, device=args.device
        )
        # Por imagen: capturamos features N veces y computamos el lens sobre cada una
        unc_per_image: dict[str, dict[str, list[float]]] = {}
        for i, (img, _) in enumerate(test_loader):
            img = adapter.preprocess(img).to(args.device)
            target = targets[i]
            # N forward passes con features capturadas
            lens_outputs_per_pass: dict[str, dict[str, list]] = {
                ln: {s: [] for s in adapter.stages} for ln in lenses
            }
            for _ in range(args.n_uncertainty_samples):
                with FeatureExtractor(adapter.model, adapter.stages) as ex:
                    _ = adapter.segment(img)
                feats = ex.features
                for lens_name, lens in lenses.items():
                    for stage in adapter.stages:
                        if hasattr(lens, "has") and not lens.has(stage):
                            continue
                        lout = lens(stage, feats[stage].to(args.device))
                        if lout is None:  # lens incompatible (e.g. LogitLens channels)
                            continue
                        logits = lout.logits
                        lens_outputs_per_pass[lens_name][stage].append(
                            dice_score(logits, target)
                        )
            for lens_name in lenses:
                unc_per_image.setdefault(lens_name, {})
                for stage in adapter.stages:
                    dices = lens_outputs_per_pass[lens_name][stage]
                    if dices:
                        std = float(np.std(dices))
                        unc_per_image[lens_name].setdefault(stage, []).append(std)
        # Añadir a la agregación
        for lens_name, stage_map in unc_per_image.items():
            for stage, std_list in stage_map.items():
                if lens_name in agg and stage in agg[lens_name]:
                    agg[lens_name][stage]["dice_uncertainty"] = std_list

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_depth_curve_csv(agg, args.output_dir / "depth_curves.csv")
    plot_depth_curves(
        agg, save_path=str(args.output_dir / "depth_curves.png"),
        show_uncertainty_bands=args.with_uncertainty,
        title=("Profundidad de decisión (con bandas de incertidumbre MC Dropout)"
               if args.with_uncertainty else "Profundidad de decisión"),
    )
    print(f"[depth-curves] OK — {args.output_dir}")


if __name__ == "__main__":
    main()
