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
    lenses = calibrate_lenses(adapter, calib_loader, n_images=args.n_calibration, device=args.device)

    # Curva de profundidad
    test_loader = DataLoader(Subset(base_ds, test_idx), batch_size=args.batch_size)
    print("[depth-curves] Ejecutando en test…")
    per_image, targets = run_depth_curves(adapter, lenses, test_loader, max_images=args.n_test)
    agg = summarize_depth_curves(per_image, targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_depth_curve_csv(agg, args.output_dir / "depth_curves.csv")
    plot_depth_curves(agg, save_path=str(args.output_dir / "depth_curves.png"))
    print(f"[depth-curves] OK — {args.output_dir}")


if __name__ == "__main__":
    main()
