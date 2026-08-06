"""``seglens-train-unet``: entrena un UNet 2D sobre LGG MRI.

Uso::

    uv run seglens-train-unet --data-root ./data/lgg --epochs 60 --batch-size 8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.metrics import dice_score, iou_score
from seglens.third_party import UNet


# ---------- augmentations ----------


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper que aplica augmentation sincronizada a imagen y máscara."""

    def __init__(self, base, indices, augment: bool = True, size=(256, 256)):
        self.base = base
        self.indices = indices
        self.augment = augment
        self.size = size
        self.geom = transforms.RandomAffine(
            degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)
        )
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        if self.augment:
            seed = int(torch.randint(0, 2**31, (1,)).item())
            torch.manual_seed(seed)
            img = self.flip(img)
            mask = self.flip(mask)
            torch.manual_seed(seed)
            img = self.vflip(img)
            mask = self.vflip(mask)
            torch.manual_seed(seed)
            img = self.geom(img)
            mask = self.geom(mask)
        return img, mask


# ---------- loss ----------


def dice_loss(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    preds = torch.sigmoid(preds)
    inter = (preds * targets).sum()
    return 1 - (2.0 * inter + smooth) / (preds.sum() + targets.sum() + smooth)


def combined_loss(preds: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    d = dice_loss(preds, targets)
    return bce_weight * bce + (1 - bce_weight) * d


# ---------- eval ----------


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        if y.dim() == 3:
            y = y.unsqueeze(1)
        out = model(x)
        loss_sum += combined_loss(out, y).item() * x.shape[0]
        for b in range(x.shape[0]):
            dice_sum += dice_score(out[b], y[b])
            iou_sum += iou_score(out[b], y[b])
        n += x.shape[0]
    model.train()
    return {
        "loss": loss_sum / max(n, 1),
        "dice": dice_sum / max(n, 1),
        "iou": iou_sum / max(n, 1),
    }


# ---------- train ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 2D UNet on LGG MRI Segmentation.")
    p.add_argument("--data-root", type=Path, required=True, help="Directorio del dataset LGG (filtrado).")
    p.add_argument("--output-dir", type=Path, default=Path("models"), help="Dónde guardar el checkpoint.")
    p.add_argument("--name", type=str, default="unet", help="Nombre del modelo (prefijo de archivos).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--init-features", type=int, default=64)
    p.add_argument("--size", type=int, default=256, help="Tamaño (HxW) al que se redimensionan las imágenes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--save-test-indices", type=Path, default=None,
                   help="Si se da, guarda los índices de test para reproducibilidad.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    size = (args.size, args.size)
    base_ds = LGGSegmentationDataset(str(args.data_root), size=size)
    train_idx, val_idx, test_idx = split_by_patient(
        base_ds.patient_ids, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"[data] {len(base_ds)} slices, {len(set(base_ds.patient_ids))} pacientes")
    print(f"       train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    if args.save_test_indices:
        from seglens.data import save_test_indices
        save_test_indices(test_idx, str(args.save_test_indices))

    train_ds = AugmentedDataset(base_ds, train_idx, augment=True, size=size)
    val_ds = Subset(base_ds, val_idx)
    test_ds = Subset(base_ds, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=3, out_channels=1, init_features=args.init_features).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"{args.name}.pth"
    log_path = args.output_dir / f"{args.name}_log.json"

    best_state = None
    best_val_dice = -1.0
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", file=sys.stdout):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True).float()
            if y.dim() == 3:
                y = y.unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = combined_loss(out, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, args.device)
        sched.step(val_metrics["dice"])
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": opt.param_groups[0]["lr"],
            "time_s": round(elapsed, 1),
        }
        history.append(log)
        print(
            f"[epoch {epoch:3d}] train_loss={train_loss:.4f}  "
            f"val_dice={val_metrics['dice']:.4f}  val_iou={val_metrics['iou']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[early-stop] patience {args.patience} agotada, parando en epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[save] {ckpt_path}")

    # Test final con el mejor modelo
    test_metrics = evaluate(model, test_loader, args.device)
    print(
        f"[test] dice={test_metrics['dice']:.4f}  iou={test_metrics['iou']:.4f}  "
        f"loss={test_metrics['loss']:.4f}"
    )
    with log_path.open("w") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "history": history,
                "best_val_dice": best_val_dice,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
