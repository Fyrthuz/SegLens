"""``seglens-run-universeg``: ejecuta UniVerSeg few-shot sobre un test set.

Uso::

    uv run seglens-run-universeg --data-root ./data/lgg --context-size 64 --channel g
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from seglens.adapter import UniVerSegAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.metrics import dice_score, iou_score


CHANNELS = {
    "rgb": None,  # promedio de 3 canales (lo hace UniVerSeg)
    "r": 0,
    "g": 1,
    "b": 2,
}


def _select_channel(img: torch.Tensor, channel: str) -> torch.Tensor:
    """Reemplaza los 3 canales de la imagen por el canal seleccionado (repetido 3x)."""
    if channel == "rgb":
        return img
    idx = CHANNELS[channel]
    g = img[idx].unsqueeze(0)  # (1, H, W)
    return g.expand(3, -1, -1).contiguous()


def _to_grayscale_3x(img: torch.Tensor) -> torch.Tensor:
    """Promedia los 3 canales y los repite 3 veces (lo que hace UniVerSeg internamente)."""
    g = img.mean(dim=0, keepdim=True)
    return g.expand(3, -1, -1).contiguous()


class ChannelSelectDataset(torch.utils.data.Dataset):
    """Wrapper que selecciona un canal específico de la imagen."""

    def __init__(self, base, indices, channel: str = "g"):
        self.base = base
        self.indices = indices
        self.channel = channel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        return _select_channel(img, self.channel), mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UniVerSeg few-shot inference on LGG MRI.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results_universeg"))
    p.add_argument("--context-size", type=int, default=64,
                   help="Número de imágenes de soporte (N).")
    p.add_argument("--channel", choices=list(CHANNELS.keys()), default="g",
                   help="Canal a usar: rgb (promedio), r, g (T1c) o b (FLAIR).")
    p.add_argument("--input-size", type=int, default=128,
                   help="Tamaño al que UniVerSeg redimensiona internamente.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=None, help="Si se da, limita el nº de imágenes de test.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # El modelo internamente redimensiona a input_size; aquí dejamos las imágenes
    # a su tamaño nativo y dejamos al wrapper redimensionar.
    base_ds = LGGSegmentationDataset(str(args.data_root), size=(args.input_size, args.input_size))
    train_idx, val_idx, test_idx = split_by_patient(
        base_ds.patient_ids, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )
    print(f"[data] {len(base_ds)} slices, {len(set(base_ds.patient_ids))} pacientes")
    print(f"       train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Contexto = primeras N imágenes de train
    if args.context_size > len(train_idx):
        print(
            f"[warn] context-size={args.context_size} > train={len(train_idx)}; "
            f"usando todos los disponibles"
        )
        ctx_indices = list(train_idx)
    else:
        ctx_indices = train_idx[: args.context_size]
    print(f"[context] {len(ctx_indices)} imágenes de soporte (channel={args.channel})")

    # Cargar adapter
    print(f"[model] Cargando UniVerSeg pretrained en {args.device}…")
    try:
        adapter = UniVerSegAdapter.from_pretrained(
            input_size=(args.input_size, args.input_size), device=args.device
        )
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Instala con: uv pip install 'seglens[universeg]'", file=sys.stderr)
        sys.exit(1)

    # Construir support set
    ctx_imgs, ctx_masks = [], []
    for i in ctx_indices:
        img, mask = base_ds[i]
        # El Adapter espera (N, 1, H, W) o (N, C, H, W) — la selección de
        # canal se hace dentro del adapter (promedio). Aquí pasamos la
        # imagen tal cual; si queremos T1c puro, lo replicamos 3x.
        if args.channel == "rgb":
            ctx_imgs.append(img)  # (3, H, W)
        else:
            ctx_imgs.append(_select_channel(img, args.channel))  # (3, H, W) con canal repetido
        ctx_masks.append(mask)  # (1, H, W)
    ctx_imgs = torch.stack(ctx_imgs, dim=0)  # (N, 3, H, W)
    ctx_masks = torch.stack(ctx_masks, dim=0)  # (N, 1, H, W)
    print(f"[context] shape imgs={tuple(ctx_imgs.shape)}  masks={tuple(ctx_masks.shape)}")
    adapter.set_context(ctx_imgs, ctx_masks)

    # Test set con selección de canal
    test_ds = ChannelSelectDataset(base_ds, test_idx, channel=args.channel)
    if args.max_test is not None:
        test_ds = Subset(test_ds, list(range(min(args.max_test, len(test_ds)))))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Inferencia y métricas
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"dice": [], "iou": []}
    sample_dirs = []
    for i, (img, gt) in enumerate(tqdm(test_loader, desc="infer")):
        img = img.to(args.device)
        gt = gt.to(args.device).float()
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        with torch.no_grad():
            logits = adapter.segment(img)  # (1, 1, 128, 128)
        # metric sobre resolución del modelo
        metrics["dice"].append(dice_score(logits[0], gt[0]))
        metrics["iou"].append(iou_score(logits[0], gt[0]))
        if args.save_predictions and i < 32:
            sample_dir = args.output_dir / f"sample_{i:04d}"
            sample_dir.mkdir(exist_ok=True)
            # guardar imágenes: original, gt, predicción binaria
            pred = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
            gt_np = (gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            orig = (img[0].mean(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(orig).save(sample_dir / "original.png")
            Image.fromarray(gt_np).save(sample_dir / "ground_truth.png")
            Image.fromarray(pred).save(sample_dir / "prediction.png")
            sample_dirs.append(str(sample_dir))

    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_test": len(metrics["dice"]),
        "test_dice_mean": float(np.mean(metrics["dice"])),
        "test_dice_std": float(np.std(metrics["dice"])),
        "test_iou_mean": float(np.mean(metrics["iou"])),
        "test_iou_std": float(np.std(metrics["iou"])),
        "sample_dirs": sample_dirs,
    }
    print(
        f"[result] test_dice={summary['test_dice_mean']:.4f}±{summary['test_dice_std']:.4f}  "
        f"test_iou={summary['test_iou_mean']:.4f}±{summary['test_iou_std']:.4f}"
    )
    log_path = args.output_dir / "summary.json"
    with log_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
