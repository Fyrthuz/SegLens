"""Smoke test end-to-end: entrena un UNet mini, calibra lentes, corre depth-curves.

Uso::

    python -m seglens.examples.smoke_unet
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from seglens.adapter import UNetAdapter
from seglens.lens import calibrate_lenses
from seglens.experiments.run_depth_curves import run_depth_curves, summarize_depth_curves
from seglens.third_party import UNet
from seglens.viz import plot_depth_curves, save_depth_curve_csv


class _RandomSegDataset(torch.utils.data.Dataset):
    def __init__(self, n=8, h=32, w=32, in_c=3):
        self.n = n
        self.h = h
        self.w = w
        self.in_c = in_c

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.randn(self.in_c, self.h, self.w)
        y = (torch.randn(1, self.h, self.w) > 0).float()
        return x, y


def main() -> None:
    print("Building mini-UNet…")
    model = UNet(in_channels=3, out_channels=1, init_features=8)
    adapter = UNetAdapter(model).eval()
    print(f"  stages: {adapter.stages}")
    print(f"  head:   {type(adapter.head()).__name__}")

    # Datos sintéticos
    train_ds = _RandomSegDataset(n=8, h=32, w=32, in_c=3)
    test_ds = _RandomSegDataset(n=4, h=32, w=32, in_c=3)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    print("\nCalibrating lenses on 4 images…")
    lenses = calibrate_lenses(adapter, train_loader, n_images=4, device="cpu")
    for name, lens in lenses.items():
        print(f"  {name}: stages = {lens.stages if hasattr(lens, 'stages') else 'N/A'}")

    print("\nRunning depth curves on test set…")
    per_image, targets = run_depth_curves(adapter, lenses, test_loader, max_images=4)
    agg = summarize_depth_curves(per_image, targets)
    for lens_name, stage_map in agg.items():
        print(f"\nLens: {lens_name}")
        for stage, ms in stage_map.items():
            dice = sum(ms["dice"]) / len(ms["dice"]) if ms["dice"] else 0.0
            print(f"  {stage:10s}  Dice = {dice:.3f}")

    print("\nSaving CSV + plot…")
    save_depth_curve_csv(agg, "results/smoke_depth_curves.csv")
    plot_depth_curves(agg, save_path="results/smoke_depth_curves.png")
    print("OK")


if __name__ == "__main__":
    main()
