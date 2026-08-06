"""``seglens-stage-uncertainty`` (E2): atribución de incertidumbre por stage.

Para cada stage ``l`` del modelo:
- Captura ``f_l`` durante N forward passes MC Dropout / TTA / Noisy.
- Calcula ``Var(f_l)`` sobre los N passes (varianza de cada feature).
- Propaga esa varianza linealmente al logit: ``Var(y)_l ≈ Var(f_l) · ‖J_l‖²``
  (simplificación: sumamos las contribuciones de los features).
- Normaliza: ``frac_l = Var(y)_l / Σ_l' Var(y)_l'``.

Hipótesis H2: la mayor contribución a la incertidumbre viene de las últimas
etapas del decoder (donde se forma la decisión).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UNetAdapter
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.hooks import FeatureExtractor
from seglens.lens import JacobianLens
from seglens.uncertainty import MCDropoutInjector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/stage_uncertainty"))
    p.add_argument("--method", choices=["mc_dropout", "tta", "noisy"], default="mc_dropout")
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--n-calibration", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--mc-p", type=float, default=0.1)
    p.add_argument("--max-pixels-jac", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-display", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
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

    # 1) Calibrar J-lens
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
    print(f"  J-lens stages: {jacobian.stages}")

    # 2) Para cada imagen de test, capturar features con MC Dropout activo
    #    y calcular varianza de features por stage.
    print("[test] capturando features con MC Dropout…")
    from seglens.uncertainty import _default_mc_stages
    mc_stages = _default_mc_stages(adapter)
    print(f"  MC Dropout stages: {mc_stages}")

    # accumulators: {stage: list of Var(f_l) per image}
    var_accum: dict[str, list[torch.Tensor]] = {s: [] for s in adapter.stages}
    pred_var_total: list[float] = []  # varianza total del logit (referencia)

    for i in test_idx:
        img, _ = base[i]
        x = img.unsqueeze(0).to(args.device)
        feats_per_pass: dict[str, list[torch.Tensor]] = {s: [] for s in adapter.stages}
        preds: list[torch.Tensor] = []
        with MCDropoutInjector(adapter.model, mc_stages, p=args.mc_p):
            for _ in range(args.n_samples):
                with FeatureExtractor(adapter.model, adapter.stages) as ex:
                    y = adapter.model(x)
                for s in adapter.stages:
                    feats_per_pass[s].append(ex.features[s].detach())
                preds.append(y.detach())
        for s in adapter.stages:
            stacked = torch.stack(feats_per_pass[s], dim=0)  # (N, C, H, W)
            var = stacked.var(dim=0, unbiased=False)  # (C, H, W)
            var_accum[s].append(var.cpu())
        # Varianza total del logit
        stacked_pred = torch.stack(preds, dim=0)
        pred_var = stacked_pred.var(dim=0, unbiased=False).mean().item()
        pred_var_total.append(pred_var)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) Por stage: media de varianza + atribución vía J_l
    print("[analysis] atribución de varianza por stage…")
    # Para cada stage, la varianza media espacial de features
    #   var_per_stage[s] = mean over pixels of mean_var_features[s]
    # Y la "magnitud del Jacobiano" ‖J_l‖² (sum de cuadrados de pesos)
    jac_norms_sq = {}
    for s in adapter.stages:
        if jacobian.has(s):
            J = jacobian.jacobian_for(s)  # (K, C_l)
            jac_norms_sq[s] = float((J ** 2).sum().item())
        else:
            jac_norms_sq[s] = 0.0

    rows = []
    for s in adapter.stages:
        # Media de varianza de features (mean over pixels & channels)
        mean_var_feat = torch.stack(var_accum[s]).mean().item()  # escalar
        # Contribución a la varianza del logit: var_feat · ‖J_l‖²
        contrib = mean_var_feat * jac_norms_sq[s]
        rows.append({
            "stage": s,
            "mean_var_features": mean_var_feat,
            "jacobian_norm_sq": jac_norms_sq[s],
            "contribution_to_logit_var": contrib,
        })

    total_contrib = sum(r["contribution_to_logit_var"] for r in rows) + 1e-12
    for r in rows:
        r["frac_of_attributed_uncertainty"] = r["contribution_to_logit_var"] / total_contrib

    # CSV
    import csv
    csv_path = args.output_dir / "stage_attribution.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> {csv_path}")

    # 4) Figura
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = [r["stage"] for r in rows]
    fracs = [r["frac_of_attributed_uncertainty"] for r in rows]
    colors = ["tab:blue" if s.startswith("down") else "tab:orange" if s.startswith("up") else "tab:gray" for s in stages]
    ax.bar(range(len(stages)), fracs, color=colors)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fracción de varianza atribuida")
    ax.set_title(f"Atribución de incertidumbre por stage ({args.method}, n={args.n_test} test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage_attribution.png", dpi=140)

    # 5) Heatmap de atribución per-píxel para N imágenes (simplificado: media espacial)
    fig, axes = plt.subplots(1, args.n_display, figsize=(3 * args.n_display, 3))
    if args.n_display == 1:
        axes = [axes]
    for k in range(args.n_display):
        # Para cada stage, calculamos el "contribution map" usando la varianza per-pixel
        # y el jacobiano (un escalar por stage en este approach simplificado)
        # attribution_map[h, w] = Σ_s (var_f_l[h,w] · ‖J_l‖²)
        # Necesitamos upsamplear todas las varianzas a un tamaño común.
        target_size = next(iter(var_accum.values()))[k].shape[-2:]  # (H, W) del primer stage
        attr_map = torch.zeros(target_size)
        for s, var_list in var_accum.items():
            v = var_list[k].mean(dim=(0, 1))  # (h, w) — media sobre batch y canales
            if v.shape != target_size:
                v = torch.nn.functional.interpolate(
                    v.unsqueeze(0).unsqueeze(0), size=target_size,
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
            attr_map += v * jac_norms_sq[s]
        attr_np = attr_map.numpy()
        # Normalizar para visualización
        if attr_np.max() > 0:
            attr_np = attr_np / attr_np.max()
        axes[k].imshow(base[test_idx[k]][0].mean(0).cpu().numpy(), cmap="gray", alpha=0.5)
        axes[k].imshow(attr_np, cmap="hot", alpha=0.6)
        axes[k].set_title(f"img {k}")
        axes[k].axis("off")
    fig.suptitle("Atribución de incertidumbre sobre imagen (rojo = alta var)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage_attribution_heatmap.png", dpi=130)

    # 6) Summary
    # H2 (relajada): las 2 últimas etapas concentran >50% de la varianza atribuida
    last_two_frac = sum(r["frac_of_attributed_uncertainty"] for r in rows[-2:])
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_test": len(test_idx),
        "stage_attribution": rows,
        "mean_pred_variance": float(np.mean(pred_var_total)),
        "h2_confirmed_last_layer_dominates": rows[-1]["frac_of_attributed_uncertainty"] > 0.5
        if rows else False,
        "h2_confirmed_last_two_layers_dominate": last_two_frac > 0.5,
        "last_two_frac_sum": last_two_frac,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== RESUMEN E2 ===")
    for r in rows:
        print(f"  {r['stage']:10s}  var_feat={r['mean_var_features']:.4e}  "
              f"‖J‖²={r['jacobian_norm_sq']:.2f}  frac={r['frac_of_attributed_uncertainty']:.3f}")
    print(f"H2 (last_two_layers_dominate, sum={last_two_frac:.3f}): "
          f"{summary['h2_confirmed_last_two_layers_dominate']}")
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
