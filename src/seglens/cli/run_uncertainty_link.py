"""``seglens-uncertainty-link`` (E1): correlación decision-depth vs incertidumbre.

Calcula, por píxel:
- ``uncertainty_map``: entropía de las predicciones bajo N forward passes MC/TTA.
- ``decision_depth``: el primer stage donde la predicción del J-lens coincide
  con ≥95% de la predicción final.

Hipótesis H1: ``ρ(Spearman, decision_depth, uncertainty) > 0`` — píxeles
decididos tarde son más inciertos.

Uso::

    uv run seglens-uncertainty-link \\
      --data-root ./data/lgg \\
      --checkpoint models/unet_real.pth \\
      --method mc_dropout \\
      --n-test 30 --n-samples 10 \\
      --output-dir results/uncertainty_link
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UNetAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.hooks import FeatureExtractor
from seglens.lens import JacobianLens
from seglens.lens.calibration import _features_to_pixels
from seglens.uncertainty import UncertaintyEstimator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/uncertainty_link"))
    p.add_argument("--method", choices=["mc_dropout", "tta", "noisy"], default="mc_dropout")
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--n-calibration", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--mc-p", type=float, default=0.1)
    p.add_argument("--max-pixels-jac", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--depth-threshold", type=float, default=0.95,
                   help="J-lens pred coincide con final si IoU >= threshold")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict_with_lens(
    adapter,
    lenses: dict[str, JacobianLens],
    x: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    """Forward + features + lens por stage. Devuelve (logits_final, lens_outputs).

    lens_outputs[stage] = Tensor(B, 1, H, W) — binarizable para comparar.
    """
    x = x.to(next(adapter.model.parameters()).device)
    with FeatureExtractor(adapter.model, adapter.stages) as ex:
        out = adapter.model(x)
    feats = ex.features
    lens_per_stage: dict[str, torch.Tensor] = {}
    for stage in adapter.stages:
        # Solo J-lens (causal). Si no está calibrada, saltamos.
        if "jacobian" in lenses and lenses["jacobian"].has(stage):
            lens_out = lenses["jacobian"](stage, feats[stage])
            lens_per_stage[stage] = lens_out.logits
    return out, lens_per_stage


@torch.no_grad()
def decision_depth_per_image(
    adapter,
    lenses: dict[str, JacobianLens],
    x: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Para cada píxel, devuelve el primer stage donde la pred J-lens
    coincide con la final (IoU binario >= threshold), o ``len(stages)`` si nunca.
    """
    final_logits, lens_per_stage = predict_with_lens(adapter, lenses, x)
    final_bin = (torch.sigmoid(final_logits[0, 0]) > 0.5).cpu().numpy()
    H, W = final_bin.shape
    depth = np.full((H, W), len(adapter.stages), dtype=np.int32)

    for s_idx, stage in enumerate(adapter.stages):
        if stage not in lens_per_stage:
            continue
        lens_bin = (torch.sigmoid(lens_per_stage[stage][0, 0]) > 0.5).cpu().numpy()
        # IoU per-pixel mask: comparamos bin por bin
        intersection = (lens_bin & final_bin).sum()
        union = (lens_bin | final_bin).sum()
        iou = intersection / max(union, 1)
        if iou >= threshold:
            # marca TODOS los píxeles como decididos en este stage
            # (es una métrica gruesa pero suficiente para el experimento)
            depth[:] = min(depth[:].max(), s_idx)
            # Para hacerlo más fino: para cada píxel, ver si la lens
            # en este stage lo predice igual que la final.
            # Versión simplificada: primer stage con iou global >= thresh
            break
    return depth


@torch.no_grad()
def decision_depth_per_pixel(
    adapter,
    lenses: dict[str, JacobianLens],
    x: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.95,
) -> np.ndarray:
    """Versión per-pixel del decision depth.

    Para cada stage, contamos en cuántos píxeles la pred J-lens coincide con la
    final. ``depth[h,w] = primer stage donde coinciden`` (o len(stages)).
    """
    final_logits, lens_per_stage = predict_with_lens(adapter, lenses, x)
    final_bin = (torch.sigmoid(final_logits[0, 0]) > 0.5).cpu().numpy()
    H, W = final_bin.shape
    depth = np.full((H, W), len(adapter.stages), dtype=np.int32)
    # agreement[h, w, s] = 1 si lens_s coincide con final en (h, w)
    for s_idx, stage in enumerate(adapter.stages):
        if stage not in lens_per_stage:
            continue
        lens_bin = (torch.sigmoid(lens_per_stage[stage][0, 0]) > 0.5).cpu().numpy()
        agree = (lens_bin == final_bin)
        # Marcar como "decided at s_idx" los píxeles que aún no están decididos
        # y que en este stage SÍ coinciden.
        undecided = depth >= len(adapter.stages)
        new_decided = undecided & agree
        depth[new_decided] = s_idx
    return depth


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = LGGSegmentationDataset(
        str(args.data_root), size=(args.size, args.size)
    )
    train_idx, _, test_idx = split_by_patient(base.patient_ids, 0.70, 0.15, args.seed)
    test_idx = test_idx[: args.n_test]
    calib_idx = train_idx[: args.n_calibration]
    print(f"[setup] train={len(train_idx)} calib={len(calib_idx)} test={len(test_idx)}")

    adapter = UNetAdapter.from_checkpoint(
        str(args.checkpoint), in_channels=3, out_channels=1, device=args.device
    )
    print(f"[adapter] stages={adapter.stages}")

    # 1) Calibrar J-lens (recalibrar siempre, según decisión 3)
    print("[calibration] J-lens…")
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_hw = (args.size, args.size)
    jacobian = JacobianLens(target_hw=target_hw, out_channels=1)
    jacobian.calibrate(
        adapter, DataLoader(Subset(base, calib_idx), batch_size=1),
        n_images=args.n_calibration, device=args.device,
        max_pixels_per_image=args.max_pixels_jac,
    )
    lenses = {"jacobian": jacobian}

    # 2) Para cada imagen de test, calcular decision_depth + uncertainty_map
    print("[test] calculando depth + uncertainty…")
    est = UncertaintyEstimator(
        adapter, n_samples=args.n_samples, mc_p=args.mc_p, device=args.device
    )
    all_uncertainty = []
    all_depth = []
    for i in test_idx:
        img, mask = base[i]
        x = img.unsqueeze(0).to(args.device)
        u = est.uncertainty_map(x, method=args.method)
        uncertainty = u["entropy"][0, 0].cpu().numpy()
        depth = decision_depth_per_pixel(adapter, lenses, x, mask, args.depth_threshold)
        all_uncertainty.append(uncertainty)
        all_depth.append(depth)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unc_arr = np.stack(all_uncertainty, axis=0)  # (N, H, W)
    dep_arr = np.stack(all_depth, axis=0)        # (N, H, W)

    # 3) Correlación per-píxel
    print("[analysis] correlación Spearman…")
    flat_unc = unc_arr.flatten()
    flat_dep = dep_arr.flatten()
    rho, pval = spearmanr(flat_unc, flat_dep)

    # Discretización: incertidumbre media en early vs late
    early_mask = flat_dep <= len(adapter.stages) // 3
    late_mask = flat_dep >= 2 * len(adapter.stages) // 3
    mean_unc_early = float(unc_arr[early_mask.reshape(unc_arr.shape)].mean()) if early_mask.any() else 0.0
    mean_unc_late = float(unc_arr[late_mask.reshape(unc_arr.shape)].mean()) if late_mask.any() else 0.0
    # Versión correcta: por-píxel a través de todas las imágenes
    unc_flat_by_pix = unc_arr.reshape(-1)
    dep_flat_by_pix = dep_arr.reshape(-1)
    n_early = (dep_flat_by_pix <= len(adapter.stages) // 3).sum()
    n_late = (dep_flat_by_pix >= 2 * len(adapter.stages) // 3).sum()
    mean_unc_early_pix = float(unc_flat_by_pix[dep_flat_by_pix <= len(adapter.stages) // 3].mean()) if n_early > 0 else 0.0
    mean_unc_late_pix = float(unc_flat_by_pix[dep_flat_by_pix >= 2 * len(adapter.stages) // 3].mean()) if n_late > 0 else 0.0

    # 4) Guardar
    np.save(args.output_dir / "uncertainty_maps.npy", unc_arr)
    np.save(args.output_dir / "decision_depth.npy", dep_arr)
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_test": len(test_idx),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_early_pixels": int(n_early),
        "n_late_pixels": int(n_late),
        "mean_uncertainty_early": mean_unc_early_pix,
        "mean_uncertainty_late": mean_unc_late_pix,
        "h1_confirmed": bool(rho > 0 and pval < 0.05),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 5) Figuras
    print("[viz] scatter + overlay…")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    # subsample para no saturar
    n_show = min(5000, flat_unc.size)
    idx = np.random.choice(flat_unc.size, n_show, replace=False)
    ax.scatter(flat_dep[idx], flat_unc[idx], alpha=0.1, s=2, c="tab:blue")
    ax.set_xlabel("Decision depth (stage index)")
    ax.set_ylabel("Uncertainty (entropy)")
    ax.set_title(f"H1: depth vs uncertainty — ρ = {rho:.3f} (p = {pval:.2e})\n"
                 f"early unc = {mean_unc_early_pix:.3f}  |  late unc = {mean_unc_late_pix:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "scatter.png", dpi=140)

    # Overlay: mostrar incertidumbre sobre la imagen original (5 ejemplos)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for k in range(min(5, len(test_idx))):
        img_np = base[test_idx[k]][0].mean(0).cpu().numpy()  # grayscale
        # overlay incertidumbre
        axes[0, k].imshow(img_np, cmap="gray")
        axes[0, k].set_title(f"img {k} (T1c)")
        axes[0, k].axis("off")
        im = axes[1, k].imshow(unc_arr[k], cmap="hot", vmin=0, vmax=unc_arr.max())
        axes[1, k].set_title(f"uncertainty (max={unc_arr[k].max():.3f})")
        axes[1, k].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label="entropy")
    fig.suptitle(f"Per-pixel uncertainty ({args.method})", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output_dir / "uncertainty_overlay.png", dpi=130)

    # Print final summary
    print(f"\n=== RESUMEN E1 ===")
    print(f"ρ(depth, uncertainty) = {rho:.4f}  (p = {pval:.2e})")
    print(f"H1 confirmada:           {summary['h1_confirmed']}")
    print(f"Unc early-decision:     {mean_unc_early_pix:.4f}")
    print(f"Unc late-decision:      {mean_unc_late_pix:.4f}")
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
