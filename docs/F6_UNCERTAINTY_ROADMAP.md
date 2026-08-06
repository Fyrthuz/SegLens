# F6 — Roadmap: Integración de Incertidumbre con SegLens

> Plan de implementación para la integración de cuantificación de incertidumbre (MC Dropout, TTA) con el pipeline de explicabilidad J-space. Versión del 2026-08-06.

---

## 0. Contexto

El pipeline actual de SegLens (F0–F5) mide **dónde** se forma la decisión de tumor en cada capa del modelo (J-lens) y **qué conceptos clínicos** están activos (CAVs sobre T1c). No cuantifica **cuánto** de esa decisión es estable bajo perturbaciones del modelo o de la entrada.

Esta fase añade:
1. Estimación de incertidumbre per-píxel con **MC Dropout** (hook-based) y **TTA** (con rotaciones donde aplica).
2. Tres experimentos que cruzan incertidumbre con el J-lens y los CAVs.
3. Bandas de incertidumbre en las curvas de profundidad ya existentes.

---

## 1. Decisiones tomadas (confirmadas)

| # | Decisión | Implicación |
|---|---|---|
| 1 | Hook-based MC Dropout (sin reentrenar) | `MCDropoutInjector` sobre `up3`/`up4` (UNet) y `dec_blocks.2.target.vmapped` (UniVerSeg). Plan B: reentrenar con `Dropout2d(p=0.1)` si la calidad es insuficiente. |
| 2 | Bandas de incertidumbre en `depth_curves.png` | `fill_between(ys - std, ys + std, alpha=0.2)` en `plot_depth_curves`. |
| 3 | Recalibrar J-lens siempre | Sin caching; cada experimento invoca `calibrate_lenses` internamente. |
| 4 | Visualizar incertidumbre como PNG | `uncertainty_overlay.png` por imagen de muestra (5-10). |
| 5 | TTA: todas las transforms aplicables (UNet: 6 incl. rotaciones con padding; UniVerSeg: hflip solo) | `_default_tta(adapter, with_rotation=True)` detecta Adapter. |

---

## 2. Fases

### U0 — Decisión de implementación MC Dropout (1h)

- [ ] Implementar `MCDropoutInjector` (hook-based)
- [ ] Smoke test: 10 passes en UNet → verificar varianza > 0 y predicciones coherentes
- [ ] Si varianza ≈ 0, reentrenar UNet con `Dropout2d(p=0.1)` en `up3`/`up4` (5 min)

### U1 — Módulo `src/seglens/uncertainty.py` (2h)

- [ ] `MCDropoutInjector` (context manager)
- [ ] `UncertaintyEstimator` con métodos: `mc_dropout`, `tta`, `noisy`
- [ ] `uncertainty_map(x, method)` → dict `{mean, std, entropy}` shape `(1, 1, H, W)`
- [ ] Helpers: `_default_mc_stages(adapter)`, `_default_tta(adapter, with_rotation)`

### U2 — Tests (1h)

- [ ] `tests/test_uncertainty.py`:
  - `MCDropoutInjector` añade varianza (10 passes dan predicciones distintas)
  - `UncertaintyEstimator.mc_dropout()` retorna shapes correctas
  - `entropy` ∈ [0, log 2]
  - `_default_tta` retorna 6 transforms para UNet, 2 para UniVerSeg
  - TTA con rotación 90° es invertible

### U3 — `adapter.uncertainty()` (1h)

- [ ] Añadir método opcional al `SegmentationAdapter` Protocol
- [ ] `UNetAdapter.uncertainty(x, method, n_samples)`
- [ ] `UniVerSegAdapter.uncertainty(x, method, n_samples)` (TTA solo, MC con hook en target branch)
- [ ] Tests en `tests/test_adapter.py`

### U4 — E1: `seglens-uncertainty-link` (3h)

- [ ] `src/seglens/cli/run_uncertainty_link.py`
- [ ] CLI: `--checkpoint`, `--data-root`, `--method`, `--n-test`, `--n-calibration`, `--n-samples`, `--output-dir`
- [ ] Calibra J-lens
- [ ] Calcula per-píxel: `uncertainty_map` + `decision_depth` (primer stage donde binarizado coincide ≥95% con final)
- [ ] `ρ(Spearman, decision_depth, uncertainty)` + recta de regresión
- [ ] Output: `uncertainty_maps.npy`, `decision_depth.npy`, `uncertainty_overlay.png`, `scatter.png`, `summary.json`
- [ ] Validar H1 (ρ > 0)

### U5 — E2: `seglens-stage-uncertainty` (3h)

- [ ] `src/seglens/cli/run_stage_uncertainty.py`
- [ ] Calibra J-lens
- [ ] Para cada imagen: `Var(f_l)` con N MC passes, `contribución_l = sqrt(Var(f_l) @ J_l^T · J_l)`, normalizar
- [ ] Output: `stage_attribution.csv`, `stage_uncertainty_heatmap.png`, `summary.json`
- [ ] Validar H2 (contribución mayor en últimas etapas)

### U6 — E3: extender `seglens-ablation` con ΔUncertainty (2h)

- [ ] `run_ablation.py`: añadir `method`, `n_samples`; calcular `ΔEntropy`
- [ ] `cli/run_ablation.py`: añadir flags `--method`, `--n-samples`
- [ ] Output: `ablation_with_uncertainty.csv` con columnas `clean_entropy, ablated_entropy, delta_entropy`
- [ ] Validar H3 (ΔEntropy ≠ 0 incluso cuando ΔDice ≈ 0)

### U7 — Bandas en `seglens-depth-curves` (2h)

- [ ] `viz.py`: `plot_depth_curves(uncertainty_bands=...)` con `fill_between`
- [ ] `aggregate_depth_curve`: acepta `dice_uncertainties` opcional
- [ ] `save_depth_curve_csv`: columnas extra `dice_uncertainty_mean`, `dice_uncertainty_std`
- [ ] `cli/run_depth_curves.py`: flag `--with-uncertainty` que ejecuta MC/TTA por stage

### U8 — Entry points + documentación (2h)

- [ ] `pyproject.toml`: 2 nuevos entry points
- [ ] `README.md`: nueva subsección "F6 — Incertidumbre" con tabla de outputs y comandos
- [ ] `ROADMAP.md`: marcar F6 como ✅
- [ ] Commit + push

---

## 3. Estructura de archivos a crear/modificar

```
src/seglens/
├── uncertainty.py                    # NUEVO
├── adapter.py                        # + uncertainty()
├── experiments/run_ablation.py       # + ΔUncertainty
├── cli/
│   ├── run_uncertainty_link.py       # NUEVO (E1)
│   ├── run_stage_uncertainty.py      # NUEVO (E2)
│   ├── run_ablation.py               # + --method, --n-samples
│   └── run_depth_curves.py           # + --with-uncertainty
└── viz.py                            # + bandas en plot

tests/
├── test_uncertainty.py               # NUEVO
└── test_adapter.py                   # + uncertainty tests

pyproject.toml                        # + 2 entry points
README.md                             # + sección F6
ROADMAP.md                            # F6 → ✅
```

---

## 4. Estimación

| Fase | Horas |
|---|---|
| U0 | 1 |
| U1 | 2 |
| U2 | 1 |
| U3 | 1 |
| U4 | 3 |
| U5 | 3 |
| U6 | 2 |
| U7 | 2 |
| U8 | 2 |
| **Total** | **~17 h** |

---

## 5. Hipótesis a validar

1. **H1 (alta confianza)**: `ρ(decision_depth, uncertainty) > 0` — píxeles decididos tarde son más inciertos.
2. **H2 (media)**: la mayor contribución a la incertidumbre viene de las últimas etapas del decoder.
3. **H3 (baja)**: ablaciones que no mueven Dice pueden reducir incertidumbre → el modelo "ve" vía CAV pero no la usa para decidir.

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Hook-based MC Dropout no inyecta varianza real | Reentrenar UNet con `Dropout2d(p=0.1)` (5 min) |
| OOM al combinar MC × J-lens | Sub-muestrear J-lens a 16 píxeles/imagen |
| TTA + UniVerSeg con rotaciones = NaN/padding | Solo hflip; documentar |
| Stage-uncertainty attribution no es causal | Documentar como "linearized variance attribution" |
| ΔEntropy ≈ 0 en todas las ablaciones | Esperado; reportar como hallazgo honesto |

---

## 7. Exit criteria

1. ✅ `pytest tests/ -v` → todos pasan (49 actuales + ~8 nuevos)
2. ✅ `seglens-uncertainty-link` produce `scatter.png` + `summary.json` con `spearman_rho`
3. ✅ `seglens-stage-uncertainty` produce `stage_attribution.csv` + heatmap
4. ✅ `seglens-ablation --with-uncertainty` produce CSV con ΔEntropy
5. ✅ `seglens-depth-curves --with-uncertainty` produce plot con bandas sombreadas
6. ✅ H1 verificada: `spearman_rho > 0`
7. ✅ README actualizado, commit + push

---

## 8. Cronograma de ejecución (esta sesión)

1. **Fase U0–U1** (3h): MC Dropout injector + módulo uncertainty
2. **Fase U2–U3** (2h): tests + adapter.uncertainty()
3. **Fase U4** (3h): E1 seglens-uncertainty-link
4. **Fase U5** (3h): E2 seglens-stage-uncertainty
5. **Fase U6** (2h): E3 extensión ablation
6. **Fase U7** (2h): bandas en depth_curves
7. **Fase U8** (2h): entry points + README + commit
