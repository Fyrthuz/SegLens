"""``seglens-ablation``: ablaciona CAVs de conceptos por etapa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UNetAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.experiments.run_concept_scoring import fit_concepts
from seglens.experiments.run_ablation import run_ablation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/ablation"))
    p.add_argument("--n-calibration", type=int, default=50)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--n-random-controls", type=int, default=5)
    p.add_argument("--stages", type=str, nargs="+", default=None,
                   help="Subset de stages a ablacionar (default: todos).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    base_ds = LGGSegmentationDataset(str(args.data_root), size=(args.size, args.size))
    train_idx, val_idx, test_idx = split_by_patient(
        base_ds.patient_ids, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )
    calib_idx = train_idx[: args.n_calibration]
    if args.n_test < len(test_idx):
        test_idx = test_idx[: args.n_test]

    adapter = UNetAdapter.from_checkpoint(
        str(args.checkpoint), in_channels=3, out_channels=1, device=args.device
    )
    if args.stages:
        adapter.stages = [s for s in adapter.stages if s in args.stages]

    print("[ablation] Ajustando CAVs…")
    calib_loader = DataLoader(Subset(base_ds, calib_idx), batch_size=1)
    cavs, thresholds = fit_concepts(adapter, calib_loader, n_images=args.n_calibration, device=args.device)

    test_loader = DataLoader(Subset(base_ds, test_idx), batch_size=1)
    print(f"[ablation] Ablacionando {len(cavs)} stages × {len(next(iter(cavs.values())))} concepts…")
    results = run_ablation(
        adapter, cavs, test_loader,
        device=args.device, strength=args.strength, n_random_controls=args.n_random_controls
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for (stage, concept), m in results.items():
        out.append({
            "stage": stage,
            "concept": concept,
            **m,
        })
    out_path = args.output_dir / "ablation.csv"
    import csv
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "concept", "clean_dice", "ablated_dice", "delta_dice", "n_images", "random_delta_dice"])
        w.writeheader()
        for row in out:
            w.writerow(row)
    summary = {
        "args": vars(args),
        "n_results": len(out),
        "results": out,
    }
    (args.output_dir / "ablation.json").write_text(json.dumps(summary, indent=2))
    print(f"[ablation] OK — {out_path}")


if __name__ == "__main__":
    main()
