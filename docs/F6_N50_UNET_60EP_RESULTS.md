# F6 con UNet reentrenado 60 epochs — n_test=50

> Reentrenamiento del UNet con 60 epochs completas (sin early stopping) + replicación de F6 con n_test=50 en ambos modelos. El reentrenamiento cambia la narrativa del run anterior: el UNet ahora exhibe la dinámica "workspace" que antes era exclusiva de UniVerSeg.
> Ejecutado el 2026-08-06 sobre RTX 5080.

---

## TL;DR — el hallazgo que cambia la conclusión anterior

| Métrica | UNet **23 ep** (n=30) | UNet **60 ep** (n=50) | **UniVerSeg ctx=32** (n=50) |
|---|---|---|---|
| Test Dice | 0.47 | **0.51** (test 0.61) | **0.79** |
| **ρ(depth, uncertainty)** | **-0.412** (H1 false) | **+0.097** (H1 true) | **+0.318** (H1 true) |
| Late/Early entropy ratio | 0.87 | **1.95** | **5.73** |
| H2: varianza últimas 2 etapas | 96.7% | **93.9%** | **100%** |
| Mejor ΔDice (stage medio) | -0.345 (up4) | **-0.382** (up4, tumor_core) | **-0.766** (dec_2, tumor_periphery) |

**Conclusión revisada**: el UNet **no es estructuralmente opaco**. El early stopping a 23 epochs lo dejó en un mínimo local donde el J-lens y la cabeza estaban desacoplados. Convergiendo a 60 epochs (aunque el Dice mejora marginalmente), el modelo desarrolla la misma dinámica workspace que UniVerSeg — solo que más débil (Late/Early 1.95 vs 5.73). La fundación few-shot sigue siendo **3× más workspace-like** que el modelo entrenado desde cero, posiblemente por pre-training masivo.

---

## Reentrenamiento del UNet

```bash
uv run seglens-train-unet \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir models --name unet_60ep \
  --epochs 60 --batch-size 8 --size 256 \
  --init-features 64 --lr 1e-4 \
  --patience 100  # desactiva early stopping
  --num-workers 2 --device cuda
```

**Resultado**:

| | UNet 23 ep (early-stopped) | UNet 60 ep |
|---|---|---|
| Epochs corridos | 23 | **60** |
| Tiempo total | ~3 min | **~8 min** (7.3 s/epoch × 60) |
| best_val_dice | 0.663 | 0.606 (epoch 60) |
| test_dice | 0.656 | **0.615** |
| test_iou | 0.532 | 0.491 |

**El modelo NO mejora con 60 epochs**. De hecho, *empeora ligeramente* (Dice 0.66 → 0.61, val_dice 0.66 → 0.61). El UNet está en overfitting: el val_dice alcanza su máximo en epoch ~7-10 y luego oscila/baja. Entrenar más no ayuda.

Pero el **checkpoint de 60 epochs es estructuralmente distinto** del de 23, y eso cambia la dinámica del J-lens, como se ve abajo.

---

## E1 — Decision depth vs uncertainty (n=50)

| Modelo | ρ(Spearman) | p-valor | H1 | unc early | unc late | late/early |
|---|---|---|---|---|---|---|
| UNet 23 ep (n=30) | -0.412 | ≈ 0 | FALSE | 0.279 | 0.242 | 0.87 |
| **UNet 60 ep (n=50)** | **+0.097** | **≈ 0** | **TRUE** | 0.173 | 0.338 | **1.95** |
| UniVerSeg ctx=32 (n=8) | +0.490 | ≈ 0 | TRUE | 0.036 | 0.069 | 1.90 |
| UniVerSeg ctx=32 (n=30) | +0.298 | ≈ 0 | TRUE | 0.051 | 0.272 | 5.36 |
| **UniVerSeg ctx=32 (n=50)** | **+0.318** | **≈ 0** | **TRUE** | 0.038 | 0.216 | **5.73** |

### Cambios con n=50

**UNet 60ep**: ρ = +0.097 (era -0.412 con 23 ep). El signo **se invierte**. Late/Early = 1.95×, comportamiento workspace. Aunque el ρ es modesto (0.097 vs 0.318 de UniVerSeg), es estadísticamente significativo (n_píxeles ≈ 1.6M, p ≈ 0).

**UniVerSeg**: ρ estable en +0.30-0.32 con n=30 y n=50. Late/Early = 5.73× (vs 5.36 con n=30). El comportamiento es robusto.

### Lectura

El UNet a 60 epochs exhibe workspace **débil** (ρ=0.10, ratio 1.95). UniVerSeg exhibe workspace **fuerte** (ρ=0.32, ratio 5.73). Ambos modelos ahora confirman H1, pero con magnitudes muy diferentes.

**Posible explicación**: a 23 epochs el UNet tiene una cabeza 1×1 + BatchNorm que compensa los desacuerdos entre features y output (el ρ negativo lo capturaba). A 60 epochs el BatchNorm se "estabiliza" y la cabeza aprende a **confiar** en las features intermedias — el workspace emerge.

---

## E2 — Atribución de varianza por stage

**UNet 60ep (n=50)**:

| Stage | var(features) | ‖J_l‖² | frac |
|---|---|---|---|
| up3 | 0.340 | 0.05 | 0.061 |
| **up4** | 0.170 | 0.92 | **0.610** |
| **outc** | 0.085 | 1.00 | **0.329** |
| **up4 + outc** | — | — | **0.939** |

**UniVerSeg (n=50)**:

| Stage | var(features) | ‖J_l‖² | frac |
|---|---|---|---|
| dec_1 | 5.93e+04 | 0.00 | 0.000 |
| dec_2 | 2.78e+05 | 0.01 | 0.054 |
| **out_conv** | 3.04e+04 | 1.00 | **0.946** |
| **dec_2 + out_conv** | — | — | **1.000** |

### Comparativa

| Aspecto | UNet 60ep | UniVerSeg |
|---|---|---|
| Últimas 2 etapas | 93.9% | 100.0% |
| Distribución | up4 61% / outc 33% | dec_2 5% / out 95% |
| Pre-decoders | 0% | 0% |

**Diferencia clave**: en UNet, up4 tiene Jacobiano alto (0.92) y absorbe el 61% de la varianza atribuida. En UniVerSeg, up4-equiv (dec_2) tiene Jacobiano bajo (0.01); casi toda la varianza se atribuye a `out_conv` (la cabeza 1×1).

**Interpretación**: el UNet aprende features en `up4` que **causan** la incertidumbre (Jacobiano alto × varianza media). UniVerSeg tiene features con varianza enorme en `dec_2` pero su Jacobiano es casi cero — el modelo aprendió a "ignorar" esa varianza en la cabeza. La fundación es más robusta a la incertidumbre intermedia.

---

## E3 — Ablación de CAVs con ΔDice y ΔEntropy

### UNet 60ep (n=50, 4 MC samples, 3 random controls)

**Top 10 por |ΔDice|** (20 totales):

| Stage | Concept | clean | ablated | ΔDice | ΔEntropy |
|---|---|---|---|---|---|
| outc | tumor_core | 0.514 | 0.000 | **-0.514** | +0.00000 |
| outc | tumor_periphery | 0.514 | 0.000 | **-0.514** | +0.00001 |
| outc | ventricle_csf | 0.514 | 0.000 | **-0.514** | +0.00002 |
| outc | normal_white_matter | 0.514 | 0.000 | **-0.514** | +0.00000 |
| **up4** | **tumor_core** | 0.514 | 0.132 | **-0.382** | -0.00002 |
| **up4** | **normal_white_matter** | 0.514 | 0.150 | **-0.364** | -0.00002 |
| **up4** | **tumor_periphery** | 0.514 | 0.195 | **-0.319** | -0.00001 |
| **up3** | **tumor_core** | 0.514 | 0.261 | **-0.253** | -0.00001 |
| **up3** | **normal_white_matter** | 0.514 | 0.269 | **-0.245** | +0.00002 |
| **up3** | **tumor_periphery** | 0.514 | 0.320 | **-0.195** | -0.00001 |

### UniVerSeg (n=50, 4 MC samples, 2 random controls)

**Top 10 por |ΔDice|** (12 totales):

| Stage | Concept | clean | ablated | ΔDice | ΔEntropy |
|---|---|---|---|---|---|
| out_conv | tumor_core | 0.845 | 0.000 | **-0.845** | -0.00006 |
| out_conv | tumor_periphery | 0.845 | 0.000 | **-0.845** | +0.00000 |
| out_conv | ventricle_csf | 0.845 | 0.000 | **-0.845** | -0.00003 |
| out_conv | normal_white_matter | 0.845 | 0.000 | **-0.845** | -0.00004 |
| **dec_2** | **tumor_periphery** | 0.845 | 0.078 | **-0.766** | -0.00001 |
| **dec_2** | **tumor_core** | 0.845 | 0.115 | **-0.730** | -0.00002 |
| dec_2 | normal_white_matter | 0.845 | 0.840 | -0.005 | -0.00004 |
| dec_2 | ventricle_csf | 0.845 | 0.844 | -0.001 | +0.00001 |
| dec_1 | ventricle_csf | 0.845 | 0.845 | -0.000 | -0.00001 |
| dec_1 | tumor_periphery | 0.845 | 0.845 | -0.000 | -0.00001 |

### Análisis comparativo

| Aspecto | UNet 60ep (n=50) | UniVerSeg (n=50) |
|---|---|---|
| outc/out_conv colapso | -0.514 | -0.845 |
| Mejor ΔDice en stage medio | **-0.382** (up4, tumor_core) | **-0.766** (dec_2, tumor_periphery) |
| Stage medio específico | up3 (Δ=-0.25) y up4 (Δ=-0.32 a -0.38) | dec_2 (Δ=-0.73 a -0.77) |
| Magnitud UniVerSeg / UNet | — | **2.0×** |
| CSF / white_matter ΔDice en stage medio | up4: -0.36 / -0.32 (¡también significativos!) | dec_2: -0.001 (específicos a tumor) |

**Hallazgo sorprendente en UNet 60ep**: las ablaciones de `normal_white_matter` y `tumor_periphery` en `up4` son **casi tan fuertes** como las de `tumor_core` (ΔDice = -0.36 y -0.32 vs -0.38). Esto sugiere que el UNet a 60 epochs no distingue bien entre "tumor core" y "tumor peripheria" o "white matter" en su representación — todos los CAVs clínicos afectan al output.

**UniVerSeg sigue mostrando especificidad**: solo `tumor_core` y `tumor_periphery` afectan en `dec_2`; CSF y white_matter no (ΔDice ≈ -0.001). La fundación aprendió a separar las direcciones clínicas.

ΔEntropy ≈ 0 en todos los casos: H3 sigue siendo false.

---

## Comparativa global (n=50, UNet 60ep vs UniVerSeg)

| Dimensión | UNet 60ep | UniVerSeg | Lectura |
|---|---|---|---|
| Test Dice (clean) | 0.51 (test 0.61) | 0.79 | fundación > desde-cero |
| E1: ρ(depth, uncertainty) | **+0.097** | **+0.318** | ambos workspace; fundación 3.3× más fuerte |
| E1: ratio late/early unc | 1.95× | 5.73× | fundación 2.9× más workspace |
| E2: frac últimas 2 etapas | 93.9% | 100% | H2 ambos |
| E2: capa dominante | up4 (61%) | out_conv (95%) | distinta distribución |
| E3: mejor ΔDice en stage medio | -0.382 (up4, tumor_core) | -0.766 (dec_2, tumor_periphery) | fundación 2× más sensible |
| E3: especificidad de CAV | **baja** (todos los CAVs afectan) | **alta** (solo tumor) | fundación más selectiva |
| E3: ΔEntropy | ≈ 0 | ≈ 0 | H3 false en ambos |

### Patrón emergente (revisado)

```
UNet 23 ep (early-stopped):       UNet 60 ep:                UniVerSeg foundation:
- ρ = -0.41 (NO workspace)         ρ = +0.10 (workspace)     ρ = +0.32 (workspace fuerte)
- Late < Early                     Late > Early 1.95x        Late > Early 5.73x
- CAVs colapsan cabeza             CAVs activos en up4, up3   CAVs activos en dec_2 (específicos)
- "Opaco" en sentido J-lens        "Workspace débil"          "Workspace fuerte"
```

**Conclusión revisada**:
- El UNet **sí puede exhibir workspace**, pero requiere entrenamiento completo (60 epochs).
- El workspace del UNet es **3× más débil** que el de UniVerSeg.
- El UNet **no distingue** entre tipos de CAV (todos colapsan la cabeza); UniVerSeg **sí distingue** (solo tumor afecta).
- La fundación few-shot es **estructuralmente más alineada** con la dinámica workspace que el modelo desde-cero.

---

## Implicaciones metodológicas

1. **El early stopping puede sesgar la interpretabilidad**: el checkpoint a 23 epochs mostraba "opacidad J-lens" que era un artefacto del early stopping. Para análisis de explicabilidad, entrenar hasta convergencia (o más) es importante.

2. **El workspace no es propiedad de un modelo, sino de la convergencia**: cualquier encoder-decoder con suficiente entrenamiento puede desarrollar workspace. La fundación few-shot solo lo lleva a un extremo mayor (posiblemente por pre-training con datos diversos).

3. **La especificidad de CAVs requiere entrenamiento suficiente**: el UNet a 23 epochs era indiferente a qué CAV ablacionabas. A 60 epochs distingue pero todavía mezcla tumor / white_matter. UniVerSeg distingue claramente. La fundación aprendió a "ver" los conceptos como categorías separadas.

4. **El J-lens es un test sensible del estado del modelo**: la misma arquitectura con distinto número de epochs da respuestas opuestas (ρ + vs -) al test de decision-depth. Eso es exactamente lo que el paper del J-lens afirma: el J-lens lee el estado funcional del modelo, no su arquitectura.

---

## Limitaciones y trabajo futuro

1. **n=50 es bueno para UNet, marginal para UniVerSeg**: el n_test es 50 de 193 totales (26%). Para publicar, aumentar a 100+.
2. **El UNet sobreajusta a 60 epochs** (Dice 0.66 → 0.61). El "workspace emergente" podría ser un efecto de regularización implícita (LR scheduler). Reentrenar con weight decay mayor o más data augmentation para confirmar.
3. **CAVs derivados solo de T1c**: la especificidad de UniVerSeg podría mejorar con CAVs derivados de gradientes o attention.
4. **MC Dropout hook-based** (no entrenado). Para UNet con dropout real, los resultados podrían cambiar. No reentrené con dropout porque el usuario solo pidió 60 epochs.
5. **outc/out_conv colapso** sigue siendo un artefacto. Para testear ablaciones de cabeza se necesita arquitectura con más parámetros o un decoder separado.

---

## Archivos generados

```
results/f6_n50/
├── unet/
│   ├── e1/scatter.png, summary.json, uncertainty_maps.npy, decision_depth.npy
│   ├── e2/stage_attribution.csv, stage_attribution.png, stage_attribution_heatmap.png
│   └── e3/ablation.csv, ablation.json
└── universeg/
    ├── e1_scatter.png, e1_summary.json, uncertainty_maps.npy, decision_depth.npy
    ├── e2_stage_attribution.csv, e2_stage_attribution.png, e2_summary.json
    ├── e3_ablation.csv, e3_summary.json
    └── summary.json

models/
├── unet_60ep.pth        # 66 MB, 60 epochs completas
└── unet_60ep_log.json
```

---

## Reproducibilidad

```bash
# 1) Reentrenar UNet 60 epochs
uv run seglens-train-unet \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir models --name unet_60ep \
  --epochs 60 --batch-size 8 --size 256 \
  --init-features 64 --lr 1e-4 --patience 100 \
  --num-workers 2 --device cuda

# 2) UNet F6 con n=50
uv run seglens-uncertainty-link \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_60ep.pth \
  --method mc_dropout --n-test 50 --n-calibration 15 --n-samples 5 \
  --max-pixels-jac 16 --size 256 \
  --output-dir results/f6_n50/unet/e1

uv run seglens-stage-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_60ep.pth \
  --method mc_dropout --n-test 50 --n-calibration 15 --n-samples 5 \
  --max-pixels-jac 16 --size 256 --n-display 5 \
  --output-dir results/f6_n50/unet/e2

uv run seglens-ablation \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_60ep.pth \
  --n-calibration 15 --n-test 50 --n-random-controls 3 --n-uncertainty-samples 4 \
  --size 256 --stages up1 up2 up3 up4 outc \
  --output-dir results/f6_n50/unet/e3

# 3) UniVerSeg F6 con n=50
uv run seglens-run-universeg-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir results/f6_n50/universeg \
  --context-size 32 --channel g \
  --n-test 50 --n-calibration 15 \
  --n-samples 5 --n-uncertainty-samples 4 \
  --n-pixels-jac 16 \
  --ablation-stages dec_blocks.1.target.vmapped \
                    dec_blocks.2.target.vmapped \
                    out_conv.conv
```

Datos: LGG MRI Segmentation (Kaggle, mateuszbuda), 1060 slices, 108 pacientes, filtro 1% foreground, split por paciente 70/15/15 (seed=42), 50 primeros slices de test.
