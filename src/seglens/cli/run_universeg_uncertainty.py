"""``seglens-run-universeg-uncertainty``: F6 sobre UniVerSeg.

Adapta los 3 experimentos de incertidumbre (E1, E2, E3) al UniVerSegAdapter.
Mismas ideas que ``run_universeg_explainability`` pero centradas en
incertidumbre:

- E1: correlación píxel-a-píxel decision-depth (J-lens) vs entropy (MC).
- E2: atribución de varianza por stage vía ``J_l``.
- E3: ablación de CAVs de T1c con ΔDice y ΔEntropy.

Limitaciones de UniVerSeg:
- TTA: solo hflip (resize interno 128×128 rompe rotaciones).
- MC Dropout: hook en target branch (``enc_blocks.*.target.vmapped``) —
  UniVerSeg no tiene dropout entrenado, así que las features dropeadas son
  una aproximación de MC Dropout bayesiano.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UniVerSegAdapter
from seglens.concepts import (
    ConceptThresholds,
    concept_scores,
    derive_concept_masks,
    fit_cavs,
)
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.lens import JacobianLens
from seglens.metrics import dice_score, iou_score
from seglens.uncertainty import MCDropoutInjector, UncertaintyEstimator
from seglens.viz import plot_depth_curves, save_depth_curve_csv


# --------------------------------------------------------------------------- #
# Hooks específicos para UniVerSeg
# --------------------------------------------------------------------------- #


@contextmanager
def universeg_feature_extractor(model, stages):
    """Hooka la rama target de cada submódulo (descarta support)."""
    captured: dict[str, Tensor] = {}
    handles = []

    def make_hook(name: str):
        def fn(_m, _i, out):
            if isinstance(out, tuple):
                out = out[0]
            captured[name] = out.detach()
        return fn

    try:
        for s in stages:
            mod = model.get_submodule(s)
            handles.append(mod.register_forward_hook(make_hook(s)))
        yield captured
    finally:
        for h in handles:
            h.remove()


UNIVERSEG_STAGES: list[str] = [
    "enc_blocks.0.target.vmapped",
    "enc_blocks.1.target.vmapped",
    "enc_blocks.2.target.vmapped",
    "enc_blocks.3.target.vmapped",
    "dec_blocks.0.target.vmapped",
    "dec_blocks.1.target.vmapped",
    "dec_blocks.2.target.vmapped",
    "out_conv.conv",
]


def select_channel(img: Tensor, channel: str) -> Tensor:
    if channel == "rgb":
        return img
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    return img[idx].unsqueeze(0).expand(3, -1, -1).contiguous()


class _ChannelDS(torch.utils.data.Dataset):
    def __init__(self, base, indices, channel):
        self.base = base
        self.indices = indices
        self.channel = channel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, mask = self.base[self.indices[i]]
        return select_channel(img, self.channel), mask


# --------------------------------------------------------------------------- #
# Calibración J-lens
# --------------------------------------------------------------------------- #


def calibrate_jacobian_universeg(adapter, calib_indices, base, channel,
                                  n_cal, n_pixels_jac, device):
    """Calibra el J-lens iterando sobre imágenes de calibración."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    target_hw = adapter.input_size
    jacobian = JacobianLens(target_hw=target_hw, out_channels=1)
    accumulators: dict = {}
    model = adapter.model

    for idx in calib_indices[:n_cal]:
        img, _ = base[idx]
        x = adapter.preprocess(select_channel(img, channel)).to(device)
        x = x.detach().clone()
        captured: dict[str, Tensor] = {}

        def make_hook(name: str):
            def fn(_m, _i, out):
                if isinstance(out, tuple):
                    out = out[0]
                captured[name] = out
            return fn

        handles = [
            model.get_submodule(s).register_forward_hook(make_hook(s))
            for s in adapter.stages
        ]
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                y = model(x, adapter._context_images, adapter._context_masks)
                B, K, H_y, W_y = y.shape
                for s_idx, stage in enumerate(adapter.stages):
                    f_l = captured[stage]
                    _, C_l, H_l, W_l = f_l.shape
                    jac = torch.zeros(K, C_l, device=f_l.device, dtype=f_l.dtype)
                    n_pix = 0
                    is_last = s_idx == len(adapter.stages) - 1
                    step = max(1, (H_l * W_l) // n_pixels_jac) if n_pixels_jac else 1
                    for hy in range(H_l):
                        for wy in range(W_l):
                            if (hy * W_l + wy) % step != 0:
                                continue
                            for k in range(K):
                                g = torch.autograd.grad(
                                    outputs=y[0, k, hy, wy],
                                    inputs=f_l,
                                    retain_graph=(not is_last) or (k < K - 1),
                                    only_inputs=True,
                                )[0]
                                jac[k] += g[0, :, hy, wy]
                                n_pix += 1
                    if n_pix > 0:
                        jac = jac / n_pix
                    if stage not in accumulators:
                        accumulators[stage] = (jac.clone(), 1)
                    else:
                        prev_jac, prev_n = accumulators[stage]
                        n_total = prev_n + 1
                        accumulators[stage] = (
                            prev_jac + (jac - prev_jac) / n_total,
                            n_total,
                        )
        finally:
            for h in handles:
                h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from seglens.lens.jacobian_lens import _Jacobian
    for stage, (jac, n) in accumulators.items():
        jacobian._jacobians[stage] = _Jacobian(weight=jac, n_pixels=n)
    return jacobian


# --------------------------------------------------------------------------- #
# E1: decision depth per-pixel
# --------------------------------------------------------------------------- #


@torch.no_grad()
def decision_depth_per_pixel_universeg(adapter, jacobian, x, final_bin):
    """Primer stage donde la pred J-lens coincide con la final en cada píxel."""
    H, W = final_bin.shape
    depth = np.full((H, W), len(adapter.stages), dtype=np.int32)
    with universeg_feature_extractor(adapter.model, adapter.stages) as ex:
        _ = adapter.segment(x)
    for s_idx, stage in enumerate(adapter.stages):
        if not jacobian.has(stage):
            continue
        lout = jacobian(stage, ex[stage])
        if lout is None:
            continue
        lens_bin = (torch.sigmoid(lout.logits[0, 0]) > 0.5).cpu().numpy()
        # upsample si difieren tamaños
        if lens_bin.shape != final_bin.shape:
            lens_bin = F.interpolate(
                torch.from_numpy(lens_bin).float().unsqueeze(0).unsqueeze(0),
                size=final_bin.shape, mode="nearest",
            ).squeeze().numpy().astype(bool)
        agree = (lens_bin == final_bin)
        undecided = depth >= len(adapter.stages)
        depth[undecided & agree] = s_idx
    return depth


# --------------------------------------------------------------------------- #
# E3: ablación con ΔEntropy
# --------------------------------------------------------------------------- #


def _entropy_from_logits(logits):
    p = torch.sigmoid(logits)
    eps = 1e-8
    h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return float(h.mean().item())


def random_direction_like(v):
    r = torch.randn_like(v)
    return r / (r.norm() + 1e-8)


@torch.no_grad()
def run_ablation_with_uncertainty_universeg(
    adapter, cavs, base, channel, n_test, n_random, n_unc_samples,
    n_unc_mc_stages, device,
):
    """Versión UniVerSeg de run_ablation: hook en la rama target + ΔEntropy."""
    results: dict = {}
    for stage, concept_cavs in cavs.items():
        for concept, cav in concept_cavs.items():
            # Sólo stages con tamaño compatible con un Conv 1x1
            clean_scores, ablated_scores, random_scores = [], [], []
            clean_unc, ablated_unc, random_unc = [], [], []
            random_dirs = [random_direction_like(cav) for _ in range(n_random)]
            random_results = [[] for _ in random_dirs]
            random_unc_results = [[] for _ in random_dirs]

            for i in range(n_test):
                img, mask = base[i]
                x = adapter.preprocess(select_channel(img, channel)).to(device)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                target = mask.to(device).float().unsqueeze(0)

                # 1) clean
                clean_y = adapter.segment(x)
                clean_scores.append(dice_score(clean_y[0], target[0]))
                if n_unc_samples > 0:
                    u = adapter.uncertainty(x, method="mc_dropout", n_samples=n_unc_samples)
                    clean_unc.append(_entropy_from_logits(u["mean"]))

                # 2) ablated
                def make_hook(direction, alpha):
                    def fn(_m, _i, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        # ablate_concept: elimina la componente del CAV
                        d = direction / (direction.norm() + 1e-8)
                        B, C, H, W = out.shape
                        flat = out.permute(0, 2, 3, 1).reshape(-1, C)
                        coeff = flat @ d
                        flat = flat - alpha * coeff.unsqueeze(-1) * d
                        return flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    return fn

                handle = adapter.model.get_submodule(stage).register_forward_hook(
                    make_hook(cav, 1.0)
                )
                try:
                    ablated_y = adapter.segment(x)
                finally:
                    handle.remove()
                ablated_scores.append(dice_score(ablated_y[0], target[0]))
                if n_unc_samples > 0:
                    u = adapter.uncertainty(x, method="mc_dropout", n_samples=n_unc_samples)
                    ablated_unc.append(_entropy_from_logits(u["mean"]))

                # 3) random controls
                for r_idx, r_dir in enumerate(random_dirs):
                    handle_r = adapter.model.get_submodule(stage).register_forward_hook(
                        make_hook(r_dir, 1.0)
                    )
                    try:
                        rand_y = adapter.segment(x)
                    finally:
                        handle_r.remove()
                    random_results[r_idx].append(dice_score(rand_y[0], target[0]))
                    if n_unc_samples > 0:
                        u = adapter.uncertainty(x, method="mc_dropout", n_samples=n_unc_samples)
                        random_unc_results[r_idx].append(_entropy_from_logits(u["mean"]))

            n = len(clean_scores)
            entry = {
                "clean_dice": sum(clean_scores) / n if n else 0.0,
                "ablated_dice": sum(ablated_scores) / n if n else 0.0,
                "delta_dice": (sum(ablated_scores) - sum(clean_scores)) / n if n else 0.0,
                "random_delta_dice": [(sum(rs) - sum(clean_scores)) / n if n else 0.0 for rs in random_results],
                "n_images": n,
            }
            if n_unc_samples > 0 and clean_unc:
                cu = sum(clean_unc) / len(clean_unc)
                au = sum(ablated_unc) / len(ablated_unc)
                ru = [sum(rr) / len(rr) if rr else 0.0 for rr in random_unc_results]
                entry["clean_entropy"] = cu
                entry["ablated_entropy"] = au
                entry["delta_entropy"] = au - cu
                entry["random_delta_entropy"] = [r - cu for r in ru]
            results[(stage, concept)] = entry
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/universeg_uncertainty"))
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--channel", choices=["rgb", "r", "g", "b"], default="g")
    p.add_argument("--n-test", type=int, default=15)
    p.add_argument("--n-calibration", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--n-uncertainty-samples", type=int, default=4)
    p.add_argument("--n-pixels-jac", type=int, default=16)
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation-stages", type=str, nargs="+",
                   default=["dec_blocks.1.target.vmapped", "dec_blocks.2.target.vmapped", "out_conv.conv"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    base = LGGSegmentationDataset(
        str(args.data_root), size=(args.input_size, args.input_size)
    )
    train_idx, _, test_idx = split_by_patient(base.patient_ids, 0.70, 0.15, args.seed)
    test_idx = test_idx[: args.n_test]
    calib_idx = train_idx[: args.n_calibration]
    print(f"[setup] train={len(train_idx)} calib={len(calib_idx)} test={len(test_idx)}")

    # 1) Adapter + context
    adapter = UniVerSegAdapter.from_pretrained(
        input_size=(args.input_size, args.input_size), device=args.device
    )
    adapter.stages = UNIVERSEG_STAGES
    print(f"[adapter] stages = {adapter.stages}")
    print(f"[adapter] head = {adapter.head()}")

    ctx_imgs, ctx_masks = [], []
    for i in train_idx[: args.context_size]:
        img, mask = base[i]
        ctx_imgs.append(select_channel(img, args.channel))
        ctx_masks.append(mask)
    ctx_imgs = torch.stack(ctx_imgs, 0)
    ctx_masks = torch.stack(ctx_masks, 0)
    adapter.set_context(ctx_imgs, ctx_masks)
    print(f"[context] ctx_size={args.context_size} channel={args.channel}")

    # 2) Calibrar J-lens
    print(f"\n=== Calibrando J-lens (n_cal={args.n_calibration}, max_pixels={args.n_pixels_jac}) ===")
    t0 = time.time()
    jacobian = calibrate_jacobian_universeg(
        adapter, calib_idx, base, args.channel,
        args.n_calibration, args.n_pixels_jac, args.device,
    )
    print(f"  J-lens calibrado en {time.time() - t0:.1f}s, stages: {jacobian.stages}")

    # ===== E1: uncertainty-link =====
    print(f"\n=== E1: decision depth vs uncertainty (n_test={args.n_test}) ===")
    est = UncertaintyEstimator(adapter, n_samples=args.n_samples, device=args.device)
    all_uncertainty = []
    all_depth = []
    final_dices = []
    t0 = time.time()
    for i in test_idx:
        img, mask = base[i]
        x = adapter.preprocess(select_channel(img, args.channel)).to(args.device)
        # 1) Incertidumbre (entropy)
        u = est.uncertainty_map(x, method="mc_dropout")
        uncertainty = u["entropy"][0, 0].cpu().numpy()
        # 2) Predicción final y Dice
        with torch.no_grad():
            final_y = adapter.segment(x)
        final_bin = (torch.sigmoid(final_y[0, 0]) > 0.5).cpu().numpy()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        target = mask.to(args.device).float().unsqueeze(0)
        final_dices.append(dice_score(final_y[0], target[0]))
        # 3) Decision depth per-pixel
        depth = decision_depth_per_pixel_universeg(adapter, jacobian, x, final_bin)
        all_uncertainty.append(uncertainty)
        all_depth.append(depth)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    e1_time = time.time() - t0
    print(f"  E1 done in {e1_time:.1f}s")

    unc_arr = np.stack(all_uncertainty, axis=0)
    dep_arr = np.stack(all_depth, axis=0)
    flat_unc = unc_arr.flatten()
    flat_dep = dep_arr.flatten()
    rho, pval = spearmanr(flat_unc, flat_dep)

    n_total = len(adapter.stages)
    early_mask = flat_dep <= n_total // 3
    late_mask = flat_dep >= 2 * n_total // 3
    mean_unc_early = float(flat_unc[early_mask].mean()) if early_mask.any() else 0.0
    mean_unc_late = float(flat_unc[late_mask].mean()) if late_mask.any() else 0.0

    e1_summary = {
        "method": "mc_dropout",
        "n_test": len(test_idx),
        "n_samples": args.n_samples,
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_early_pixels": int(early_mask.sum()),
        "n_late_pixels": int(late_mask.sum()),
        "mean_uncertainty_early": mean_unc_early,
        "mean_uncertainty_late": mean_unc_late,
        "mean_test_dice": float(np.mean(final_dices)),
        "h1_confirmed": bool(rho > 0 and pval < 0.05),
    }
    np.save(args.output_dir / "uncertainty_maps.npy", unc_arr)
    np.save(args.output_dir / "decision_depth.npy", dep_arr)
    (args.output_dir / "e1_summary.json").write_text(json.dumps(e1_summary, indent=2))
    print(f"  ρ(depth, uncertainty) = {rho:.4f}  (p = {pval:.2e})")
    print(f"  H1 confirmada:           {e1_summary['h1_confirmed']}")
    print(f"  unc early-decision = {mean_unc_early:.4f}")
    print(f"  unc late-decision  = {mean_unc_late:.4f}")

    # Figura E1: scatter
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    n_show = min(5000, flat_unc.size)
    idx = np.random.choice(flat_unc.size, n_show, replace=False)
    ax.scatter(flat_dep[idx], flat_unc[idx], alpha=0.15, s=4, c="tab:blue",
               label=f"pixels (n={flat_unc.size:,})")
    ax.set_xlabel("Decision depth (stage index)")
    ax.set_ylabel("Uncertainty (entropy)")
    ax.set_title(f"UniVerSeg E1: depth vs uncertainty — ρ = {rho:.3f} (p = {pval:.2e})\n"
                 f"early unc = {mean_unc_early:.3f}  |  late unc = {mean_unc_late:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "e1_scatter.png", dpi=140)

    # ===== E2: stage uncertainty =====
    print(f"\n=== E2: stage-wise uncertainty attribution ===")
    t0 = time.time()
    mc_stages = ["dec_blocks.2.target.vmapped", "dec_blocks.1.target.vmapped"]
    var_accum: dict[str, list[Tensor]] = {s: [] for s in adapter.stages}
    pred_var_total = []
    for i in test_idx:
        img, _ = base[i]
        x = adapter.preprocess(select_channel(img, args.channel)).to(args.device)
        feats_per_pass: dict[str, list[Tensor]] = {s: [] for s in adapter.stages}
        preds = []
        with MCDropoutInjector(adapter.model, mc_stages, p=0.1):
            for _ in range(args.n_samples):
                with universeg_feature_extractor(adapter.model, adapter.stages) as ex:
                    y = adapter.segment(x)
                for s in adapter.stages:
                    feats_per_pass[s].append(ex[s].detach())
                preds.append(y.detach())
        for s in adapter.stages:
            stacked = torch.stack(feats_per_pass[s], dim=0)
            var_accum[s].append(stacked.var(dim=0, unbiased=False).cpu())
        pred_var_total.append(torch.stack(preds).var(dim=0, unbiased=False).mean().item())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    jac_norms_sq = {}
    for s in adapter.stages:
        if jacobian.has(s):
            J = jacobian.jacobian_for(s)
            jac_norms_sq[s] = float((J ** 2).sum().item())
        else:
            jac_norms_sq[s] = 0.0

    rows = []
    for s in adapter.stages:
        mean_var = torch.stack(var_accum[s]).mean().item()
        contrib = mean_var * jac_norms_sq[s]
        rows.append({
            "stage": s,
            "mean_var_features": mean_var,
            "jacobian_norm_sq": jac_norms_sq[s],
            "contribution_to_logit_var": contrib,
        })
    total = sum(r["contribution_to_logit_var"] for r in rows) + 1e-12
    for r in rows:
        r["frac_of_attributed_uncertainty"] = r["contribution_to_logit_var"] / total
    e2_time = time.time() - t0
    print(f"  E2 done in {e2_time:.1f}s")

    import csv
    with (args.output_dir / "e2_stage_attribution.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Figura E2
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = [r["stage"] for r in rows]
    fracs = [r["frac_of_attributed_uncertainty"] for r in rows]
    colors = ["tab:blue" if s.startswith("enc") else "tab:orange" if s.startswith("dec") else "tab:gray" for s in stages]
    ax.bar(range(len(stages)), fracs, color=colors)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace(".target.vmapped", "").replace("blocks.", "_") for s in stages],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fracción de varianza atribuida")
    ax.set_title("UniVerSeg E2: atribución de incertidumbre por stage")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "e2_stage_attribution.png", dpi=140)

    last_two_frac = sum(r["frac_of_attributed_uncertainty"] for r in rows[-2:])
    for r in rows:
        print(f"  {r['stage']:42s}  var={r['mean_var_features']:.3e}  "
              f"‖J‖²={r['jacobian_norm_sq']:.2f}  frac={r['frac_of_attributed_uncertainty']:.3f}")
    print(f"  last_two_sum = {last_two_frac:.3f} → H2 (last two dominate): "
          f"{last_two_frac > 0.5}")

    e2_summary = {
        "n_test": len(test_idx),
        "n_samples": args.n_samples,
        "mc_stages": mc_stages,
        "stage_attribution": rows,
        "last_two_frac_sum": last_two_frac,
        "h2_confirmed": last_two_frac > 0.5,
    }
    (args.output_dir / "e2_summary.json").write_text(json.dumps(e2_summary, indent=2))

    # ===== E3: ablation with uncertainty =====
    print(f"\n=== E3: ablation with ΔEntropy (stages: {args.ablation_stages}) ===")
    # Ajustar CAVs solo para los stages que vamos a ablacionar
    print("  Ajustando CAVs (T1c, todos los stages)...")
    t0 = time.time()
    cavs: dict[str, dict[str, Tensor]] = {}
    for idx in calib_idx:
        img, mask = base[idx]
        img_ch = select_channel(img, args.channel)
        x = adapter.preprocess(img_ch).to(args.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        target = mask.to(args.device).float()
        with universeg_feature_extractor(adapter.model, adapter.stages) as ex:
            with torch.no_grad():
                _ = adapter.segment(x)
        # Reconstruir imagen RGB para que derive_concept_masks use T1c (canal 1)
        img_rgb = torch.zeros(1, 3, *img_ch.shape[-2:], device=img_ch.device)
        img_rgb[:, 1] = img_ch[0]  # el canal 0 de img_ch es el canal g replicado
        if not hasattr(main, "_thresholds"):
            # Calcular umbrales una vez
            all_imgs, all_masks = [], []
            for j, idx2 in enumerate(calib_idx):
                if j >= args.n_calibration:
                    break
                im, mk = base[idx2]
                im_ch = select_channel(im, args.channel)
                im_rgb2 = torch.zeros(3, *im_ch.shape[-2:])
                im_rgb2[1] = im_ch[0]
                all_imgs.append(im_rgb2)
                all_masks.append(mk)
            all_imgs = torch.stack(all_imgs, 0)
            all_masks = torch.stack(all_masks, 0).float()
            if all_masks.dim() == 3:
                all_masks = all_masks.unsqueeze(1)
            main._thresholds = ConceptThresholds.from_batch(all_imgs, all_masks, t1c_channel=1)
        concept_masks = derive_concept_masks(img_rgb, mask, main._thresholds, t1c_channel=1)
        for stage in adapter.stages:
            cavs_stage = fit_cavs(ex, concept_masks, stage)
            if stage not in cavs:
                cavs[stage] = cavs_stage
    thresholds = main._thresholds
    print(f"  CAVs ajustados en {time.time() - t0:.1f}s")

    # Ablación restringida a los stages solicitados
    cavs_abl = {s: cavs.get(s) for s in args.ablation_stages if s in cavs}
    for s in args.ablation_stages:
        if s not in cavs_abl and cavs:
            cavs_abl[s] = next(iter(cavs.values()))  # fallback

    t0 = time.time()
    abl_results = run_ablation_with_uncertainty_universeg(
        adapter, cavs_abl, base, args.channel, args.n_test,
        n_random=2, n_unc_samples=args.n_uncertainty_samples,
        n_unc_mc_stages=mc_stages, device=args.device,
    )
    e3_time = time.time() - t0
    print(f"  E3 done in {e3_time:.1f}s")

    rows3 = []
    for (stage, concept), m in abl_results.items():
        rows3.append({"stage": stage, "concept": concept, **m})
    with (args.output_dir / "e3_ablation.csv").open("w", newline="") as f:
        has_unc = "delta_entropy" in rows3[0] if rows3 else False
        fieldnames = ["stage", "concept", "clean_dice", "ablated_dice", "delta_dice", "n_images", "random_delta_dice"]
        if has_unc:
            fieldnames += ["clean_entropy", "ablated_entropy", "delta_entropy", "random_delta_entropy"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows3:
            w.writerow(r)

    print("  Top 5 por |ΔDice|:")
    rows3_sorted = sorted(rows3, key=lambda r: r["delta_dice"])
    for r in rows3_sorted[:5]:
        d_e = r.get("delta_entropy", 0.0)
        print(f"    {r['stage']:42s}  {r['concept']:22s}  ΔDice={r['delta_dice']:+.4f}  "
              f"ΔEntropy={d_e:+.5f}")

    e3_summary = {
        "n_test": args.n_test,
        "n_uncertainty_samples": args.n_uncertainty_samples,
        "ablation_stages": args.ablation_stages,
        "results": rows3,
    }
    (args.output_dir / "e3_summary.json").write_text(json.dumps(e3_summary, indent=2))

    # ===== Resumen global =====
    total_time = time.time() - t_start
    global_summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "e1": e1_summary,
        "e2": e2_summary,
        "e3": e3_summary,
        "total_time_s": round(total_time, 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(global_summary, indent=2))

    print(f"\n=== RESUMEN FINAL ({total_time:.1f}s) ===")
    print(f"E1  ρ(depth, uncertainty) = {rho:.4f}  H1: {e1_summary['h1_confirmed']}")
    print(f"E2  last_two_sum = {last_two_frac:.3f}  H2: {e2_summary['h2_confirmed']}")
    print(f"E3  ablation: {len(rows3)} (stage, concept) pairs, all_deltas computed")
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
