"""Pipeline de explicabilidad J-space para UniVerSeg.

UniVerSeg tiene una estructura peculiar: cada ``CrossBlock`` opera sobre
dos tensores (target, support) y devuelve una tupla. Para hacer J-lens,
hookeamos la rama ``target`` de cada bloque, que es donde se procesa la
imagen de consulta.

Stages introspectadas (target branch):
    - enc_blocks.{0..3}.target.vmapped  (4 encoder blocks)
    - dec_blocks.{0..2}.target.vmapped  (3 decoder blocks)
    - out_conv.conv                       (cabeza 1x1)
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from seglens.adapter import UniVerSegAdapter
from seglens.concepts import concept_scores
from seglens.data import LGGSegmentationDataset, split_by_patient
from seglens.experiments.run_ablation import run_ablation
from seglens.lens import JacobianLens, LogitLens, TunedLens
from seglens.lens.calibration import _features_to_pixels
from seglens.metrics import dice_score, iou_score
from seglens.viz import plot_depth_curves, save_depth_curve_csv


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


@contextmanager
def universeg_feature_extractor(model, stages):
    """Hooka cada submódulo y guarda la salida (extrayendo target si es tupla)."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/universeg_explainability"))
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--channel", choices=["rgb", "r", "g", "b"], default="g")
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--n-calibration", type=int, default=10)
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stages", type=str, nargs="+", default=UNIVERSEG_STAGES)
    p.add_argument("--max-pixels-jac", type=int, default=16)
    p.add_argument("--n-random-controls", type=int, default=3)
    return p.parse_args()


def calibrate_lenses_universeg(adapter, calib_loader, n_cal, target_hw, device, max_pixels_jac):
    """Calibra logit + tuned + jacobian lens sobre los stages de UniVerSeg."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    head = adapter.head()
    out_channels = 1
    stages = adapter.stages

    logit_lens = LogitLens(head, target_hw=target_hw)
    tuned_lens = TunedLens(target_hw=target_hw, out_channels=out_channels)
    accum_feats: dict[str, list[Tensor]] = {s: [] for s in stages}
    accum_tgts: list[Tensor] = []
    seen = 0
    for img, _ in calib_loader:
        if seen >= n_cal:
            break
        img = img.to(device)
        with universeg_feature_extractor(adapter.model, stages) as ex:
            with torch.no_grad():
                y = adapter.segment(img)
        flat = _features_to_pixels(ex, target_hw)
        for s in stages:
            accum_feats[s].append(flat[s])
        y_up = y
        if tuple(y.shape[-2:]) != target_hw:
            y_up = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        B, K, H, W = y_up.shape
        accum_tgts.append(y_up.permute(0, 2, 3, 1).reshape(-1, K).detach())
        seen += 1
    for s in stages:
        if accum_feats[s]:
            tuned_lens.fit({s: torch.cat(accum_feats[s], 0)}, torch.cat(accum_tgts, 0))

    # J-lens: per-pixel sobre la feature RAW (sin upsample)
    jac_lens = JacobianLens(target_hw=target_hw, out_channels=out_channels)
    accumulators: dict = {}
    model = adapter.model

    def _calibrate_one(img: Tensor):
        # Preprocesar igual que el Adapter: 3 canales → 1 canal grayscale
        x = adapter.preprocess(img).detach().clone()
        captured: dict[str, Tensor] = {}

        def make_hook(name: str):
            def fn(_m, _i, out):
                if isinstance(out, tuple):
                    out = out[0]
                captured[name] = out
            return fn

        handles = [
            model.get_submodule(s).register_forward_hook(make_hook(s))
            for s in stages
        ]
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                y = model(x, adapter._context_images, adapter._context_masks)
                B, K, H_y, W_y = y.shape
                for s_idx, stage in enumerate(stages):
                    f_l = captured[stage]
                    _, C_l, H_l, W_l = f_l.shape
                    jac = torch.zeros(K, C_l, device=f_l.device, dtype=f_l.dtype)
                    n_pix = 0
                    is_last = s_idx == len(stages) - 1
                    step = max(1, (H_l * W_l) // max_pixels_jac) if max_pixels_jac else 1
                    for hy in range(H_l):
                        for wy in range(W_l):
                            if (hy * W_l + wy) % step != 0:
                                continue
                            for k in range(K):
                                # retain_graph=True hasta el último pixel/k del último stage
                                last_call = (
                                    is_last
                                    and k == K - 1
                                    and (hy * W_l + wy) == (H_l * W_l - step or H_l * W_l - 1)
                                )
                                g = torch.autograd.grad(
                                    outputs=y[0, k, hy, wy],
                                    inputs=f_l,
                                    retain_graph=not last_call,
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

    for img, _ in calib_loader:
        if len(accumulators) >= 1 and all(
            accumulators[s][1] >= n_cal for s in stages
        ):
            break
        img = img.to(device)
        _calibrate_one(img)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from seglens.lens.jacobian_lens import _Jacobian
    for stage, (jac, n) in accumulators.items():
        jac_lens._jacobians[stage] = _Jacobian(weight=jac, n_pixels=n)
    return {"logit": logit_lens, "tuned": tuned_lens, "jacobian": jac_lens}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = LGGSegmentationDataset(
        str(args.data_root), size=(args.input_size, args.input_size)
    )
    train_idx, _, test_idx = split_by_patient(base.patient_ids, 0.70, 0.15, args.seed)
    test_idx = test_idx[: args.n_test]
    calib_idx = train_idx[: args.n_calibration]
    print(f"[setup] train={len(train_idx)} calib={len(calib_idx)} test={len(test_idx)}")

    adapter = UniVerSegAdapter.from_pretrained(
        input_size=(args.input_size, args.input_size), device=args.device
    )
    adapter.stages = args.stages
    print(f"[adapter] head: {adapter.head()}")

    ctx_imgs, ctx_masks = [], []
    for i in train_idx[: args.context_size]:
        img, mask = base[i]
        ctx_imgs.append(select_channel(img, args.channel))
        ctx_masks.append(mask)
    adapter.set_context(torch.stack(ctx_imgs, 0), torch.stack(ctx_masks, 0))
    print(f"[context] ctx_size={args.context_size} channel={args.channel}")

    # ----------- A) Depth curves -----------
    print("\n=== A) Depth curves (3 lenses) ===")
    target_hw = (args.input_size, args.input_size)
    calib_loader = DataLoader(_ChannelDS(base, calib_idx, args.channel), batch_size=1)
    lenses = calibrate_lenses_universeg(
        adapter, calib_loader, args.n_calibration, target_hw, args.device, args.max_pixels_jac
    )

    test_loader = DataLoader(_ChannelDS(base, test_idx, args.channel), batch_size=1)
    per_image: list = []
    targets: list[Tensor] = []
    for img, mask in test_loader:
        img = img.to(args.device)
        with universeg_feature_extractor(adapter.model, adapter.stages) as ex:
            with torch.no_grad():
                _ = adapter.segment(img)
        feats = ex
        img_results: dict = {}
        for lens_name, lens in lenses.items():
            img_results[lens_name] = {}
            for stage in adapter.stages:
                if hasattr(lens, "has") and not lens.has(stage):
                    img_results[lens_name][stage] = None
                    continue
                img_results[lens_name][stage] = lens(stage, feats[stage].to(args.device))
        per_image.append(img_results)
        targets.append(mask.cuda().float().unsqueeze(0))

    from seglens.viz import aggregate_depth_curve
    agg = aggregate_depth_curve(per_image, targets)
    save_depth_curve_csv(agg, args.output_dir / "depth_curves.csv")
    plot_depth_curves(agg, save_path=str(args.output_dir / "depth_curves.png"))
    print(f"  -> {args.output_dir / 'depth_curves.{csv,png}'}")
    for lens_name, stage_map in agg.items():
        non_empty = {s: m for s, m in stage_map.items() if m["dice"]}
        if not non_empty:
            continue
        best_s = max(non_empty, key=lambda s: sum(non_empty[s]["dice"]) / len(non_empty[s]["dice"]))
        d = sum(non_empty[best_s]["dice"]) / len(non_empty[best_s]["dice"])
        print(f"  {lens_name:10s}  best = {best_s:38s}  Dice = {d:.4f}")

    # ----------- B) Concept scoring (T1c) -----------
    print("\n=== B) Concept scoring (T1c) ===")
    # Ajuste de CAVs por etapa
    cavs: dict[str, dict[str, Tensor]] = {}
    for img, mask in calib_loader:
        img = img.to(args.device)
        mask = mask.to(args.device).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        with universeg_feature_extractor(adapter.model, adapter.stages) as ex:
            with torch.no_grad():
                _ = adapter.segment(img)
        # Concept masks usando T1c (canal g de la imagen original) — pasamos imagen RGB para que
        # derive_concept_masks use el canal g. Truco: reconstruir imagen RGB.
        img_rgb = torch.zeros(1, 3, *img.shape[-2:], device=img.device)
        img_rgb[:, 1] = img[:, 0]  # ya que el adapter reduce a gris, g == el único canal
        # Derivamos máscaras a tamaño del target
        from seglens.concepts import ConceptThresholds, derive_concept_masks, fit_cavs
        if not hasattr(main, "_thresholds"):
            # calcular umbrales una vez
            calib_imgs, calib_masks = [], []
            for j, (im, mk) in enumerate(calib_loader):
                if j >= args.n_calibration:
                    break
                im_rgb = torch.zeros_like(im)
                im_rgb[:, 1] = im[:, 0]
                calib_imgs.append(im_rgb)
                calib_masks.append(mk)
            all_imgs = torch.cat(calib_imgs, 0)
            all_masks = torch.cat(calib_masks, 0).float()
            if all_masks.dim() == 3:
                all_masks = all_masks.unsqueeze(1)
            main._thresholds = ConceptThresholds.from_batch(all_imgs, all_masks, t1c_channel=1)
        concept_masks = derive_concept_masks(img_rgb, mask, main._thresholds, t1c_channel=1)
        for stage in adapter.stages:
            cavs_stage = fit_cavs(ex, concept_masks, stage)
            if stage not in cavs:
                cavs[stage] = cavs_stage

    cavs_path = args.output_dir / "cavs.pt"
    torch.save(
        {"cavs": cavs, "thresholds": main._thresholds.fg_percentiles.tolist()},
        cavs_path,
    )
    print(f"  -> {cavs_path} ({len(cavs)} stages × {len(next(iter(cavs.values())))} concepts)")

    # ----------- C) Ablation -----------
    print("\n=== C) Ablation ===")
    abl_stages = [s for s in adapter.stages if "dec_blocks" in s or "out_conv" in s]
    adapter.stages = abl_stages
    print(f"  stages: {abl_stages}")
    # Reconfigurar CAVs a las stages que vamos a ablacionar
    cavs_abl = {s: cavs[s] for s in abl_stages if s in cavs}
    if cavs_abl:
        # Para stages que no estén en cavs, copiar las del último encoder
        for s in abl_stages:
            if s not in cavs_abl:
                fallback = next(iter(cavs_abl.values()))
                cavs_abl[s] = fallback
        results = run_ablation(
            adapter, cavs_abl, test_loader,
            device=args.device, strength=1.0, n_random_controls=args.n_random_controls,
        )
        rows = []
        for (stage, concept), m in results.items():
            rows.append({
                "stage": stage,
                "concept": concept,
                "clean_dice": m["clean_dice"],
                "ablated_dice": m["ablated_dice"],
                "delta_dice": m["delta_dice"],
                "random_delta_mean": float(np.mean(m["random_delta_dice"])) if m["random_delta_dice"] else 0.0,
            })
        import csv
        abl_csv = args.output_dir / "ablation.csv"
        with abl_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stage", "concept", "clean_dice", "ablated_dice", "delta_dice", "random_delta_mean"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  -> {abl_csv}")
        if rows:
            most_important = min(rows, key=lambda r: r["delta_dice"])
            least_important = max(rows, key=lambda r: r["delta_dice"])
            print(f"  CAV más importante: {most_important['stage']} / {most_important['concept']}  ΔDice={most_important['delta_dice']:+.4f}")
            print(f"  CAV menos importante: {least_important['stage']} / {least_important['concept']}  ΔDice={least_important['delta_dice']:+.4f}")

    # ----------- summary -----------
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_test": args.n_test,
        "n_calibration": args.n_calibration,
        "context_size": args.context_size,
        "channel": args.channel,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
