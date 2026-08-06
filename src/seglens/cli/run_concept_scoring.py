"""``seglens-concept-scoring``: genera mapas de score por concepto y etapa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UNetAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.experiments.run_concept_scoring import fit_concepts, score_concepts_on_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/concept_scoring"))
    p.add_argument("--n-calibration", type=int, default=50)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    base_ds = LGGSegmentationDataset(str(args.data_root), size=(args.size, args.size))
    train_idx, val_idx, test_idx = split_by_patient(
        base_ds.patient_ids, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )
    calib_idx = train_idx[: args.n_calibration]
    test_idx = test_idx[: args.n_test]

    adapter = UNetAdapter.from_checkpoint(
        str(args.checkpoint), in_channels=3, out_channels=1, device=args.device
    )

    calib_loader = DataLoader(Subset(base_ds, calib_idx), batch_size=1)
    print("[concepts] Ajustando CAVs…")
    cavs, thresholds = fit_concepts(adapter, calib_loader, n_images=args.n_calibration, device=args.device)

    test_loader = DataLoader(Subset(base_ds, test_idx), batch_size=1)
    print("[concepts] Scoring test set…")
    scores = score_concepts_on_test(adapter, cavs, test_loader, device=args.device, max_images=args.n_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Guardar como torch.save
    torch.save({"scores": scores, "cavs": cavs, "thresholds": thresholds.fg_percentiles.tolist()}, args.output_dir / "concept_scores.pt")
    # Resumen JSON
    summary = {
        "args": vars(args),
        "n_stages": len(scores),
        "n_concepts": len(next(iter(scores.values()))) if scores else 0,
        "n_test": args.n_test,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[concepts] OK — {args.output_dir}")


if __name__ == "__main__":
    main()
