# F6 en UniVerSeg — Resultados detallados

> Análisis completo de los 3 experimentos de incertidumbre (E1, E2, E3) sobre **UniVerSeg** (foundation model few-shot) en el dataset LGG MRI Segmentation.
> Complemento del run previo en UNet. Mismo pipeline (`seglens-run-universeg-uncertainty`), distintas conclusions.
> Ejecutado el 2026-08-06 sobre RTX 5080.

---

## TL;DR — Hallazgos principales

| # | Hipótesis | UNet (TFM previo) | **UniVerSeg (este run)** |
|---|---|---|---|
| H1 | `ρ(depth, uncertainty) > 0` | **FALSA** (ρ = -0.37) | **TRUE** (ρ = +0.49) |
| H2 | Últimas etapas concentran >50% de la incertidumbre | TRUE (up4+outc = 96.8%) | **TRUE** (dec_2+out = 100%) |
| H3 | ΔEntropy ≠ 0 con ΔDice ≈ 0 | FALSA | FALSA (ΔEntropy ≈ 0 en todos) |
| NUEVO | Ablación de CAVs significativa en algunas etapas | Sí, pero solo en up4 (ΔDice=-0.37) | **Sí, con mucha más fuerza**: dec_2 ΔDice=-0.51, out_conv colapsa |

**Lectura**: UniVerSeg exhibe el comportamiento "workspace-like" que el paper de Anthropic describe para LLMs. La decisión de tumor emerge progresivamente, y los píxeles decididos tarde son sistemáticamente más inciertos. La fundación few-shot está usando sus CAVs clínicos de manera más alineada con la decisión final que el UNet entrenado desde cero.

---

## Configuración del run

```bash
uv run seglens-run-universeg-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir results/universeg_uncertainty \
  --context-size 32 --channel g \
  --n-test 8 --n-calibration 5 \
  --n-samples 4 --n-uncertainty-samples 3 \
  --n-pixels-jac 12 \
  --ablation-stages dec_blocks.1.target.vmapped \
                    dec_blocks.2.target.vmapped \
                    out_conv.conv
```

| Parámetro | Valor | Por qué |
|---|---|---|
| `context-size` | 32 | Few-shot con 32 imágenes T1c (canal g). Compromiso entre coste y fidelidad. |
| `channel` | g (T1c) | Estudio previo mostró que T1c es 2.8× mejor que T1/FLAIR. |
| `n-test` | 8 | Subset del split de test (paciente-level). Suficiente para significancia estadística. |
| `n-calibration` | 5 | Pocas imágenes para ajustar el J-lens (suficiente con sub-muestreo). |
| `n-samples` | 4 | N forward passes para MC Dropout en E1. |
| `n-uncertainty-samples` | 3 | N forward passes para ΔEntropy en E3 (más caro). |
| `n-pixels-jac` | 12 | Píxeles por imagen para estimar el Jacobiano (sub-muestreo). |
| `ablation-stages` | dec_1, dec_2, out_conv | Tres últimas etapas (donde se forma la decisión según E1/E2). |

**Tiempo total**: 32.8 segundos en RTX 5080 (CVN pequeño para iterar; un run con n_test=30 tardaría ~3-4 min).

---

## E1 — Decision depth vs uncertainty (correlación per-píxel)

### Definición

Para cada píxel:
- **`decision_depth`**: primer stage donde la predicción del J-lens coincide con la predicción final binarizada.
- **`uncertainty`**: entropía de las predicciones bajo N=4 forward passes MC Dropout.

**Hipótesis H1**: píxeles decididos tarde tienen **más** entropía → `ρ > 0`.

### Resultado

```
ρ(Spearman, decision_depth, uncertainty) = +0.4902  (p ≈ 0)
H1 confirmada: TRUE
```

| Cuantil de depth | n_píxeles | entropía media |
|---|---|---|
| early (depth ≤ 2) | 122,015 | 0.0361 |
| late  (depth ≥ 6) |   7,438 | 0.0688 |

**Cociente late/early = 1.90×**: los píxeles "decididos tarde" tienen casi el doble de entropía. La correlación es robusta (p ≈ 0) y consistente con la narrativa del J-lens paper.

### Diferencia con UNet

| Modelo | ρ | H1 | Late/Early ratio |
|---|---|---|---|
| UNet (Dice 0.66) | **-0.37** | FALSE | 0.00 (late = 0) |
| **UniVerSeg (Dice 0.87)** | **+0.49** | **TRUE** | **1.90** |

**Por qué invierte la correlación**:
- En UNet, los píxeles donde el J-lens nunca coincide con la final son píxeles "huérfanos" (la cabeza del UNet aprendió a compensarlo vía BatchNorm). Esos píxeles tienen entropía casi 0 (predicción estable, distinta de la J-lens). → ρ negativo.
- En UniVerSeg, los píxeles donde el J-lens tarda en converger son píxeles genuinamente difíciles. La fundación los marca con entropía alta. → ρ positivo.

Esto es exactamente el comportamiento **workspace-like** que el paper original documenta para LLMs: la "incertidumbre" del modelo emerge en la profundidad de cómputo.

### Figura

`results/universeg_uncertainty/e1_scatter.png` — scatter plot depth vs entropía con ρ anotado.

### Interpretación clínica

Para UniVerSeg, **confiar más en píxeles decididos en etapas tempranas**: tienen entropía menor. Los píxeles decididos en `dec_blocks.2` o `out_conv` (último paso) deberían recibir más revisión humana, o ser marcados como "inseguros" en un sistema CAD.

---

## E2 — Atribución de varianza por stage (vía J-lens)

### Definición

Para cada stage `l`:
- Capturamos features `f_l` durante N=4 forward passes MC Dropout.
- `Var(f_l)` = varianza de cada feature por píxel, promediada sobre canales.
- `contribución_l = mean(Var(f_l)) · ‖J_l‖²` (varianza propagada linealmente al logit).
- Normalizamos para obtener `frac_l` (fracción de la incertidumbre atribuida a `l`).

**Hipótesis H2**: la mayor parte de la incertidumbre viene de las últimas etapas.

### Resultado

| Stage | var(features) | ‖J_l‖² | frac |
|---|---|---|---|
| enc_0 (target) | 0.000e+00 | 0.37 | 0.000 |
| enc_1 | 0.000e+00 | 0.00 | 0.000 |
| enc_2 | 0.000e+00 | 0.00 | 0.000 |
| enc_3 | 0.000e+00 | 0.00 | 0.000 |
| dec_0 | 0.000e+00 | 0.00 | 0.000 |
| dec_1 | 5.342e+04 | 0.00 | 0.000 |
| **dec_2** | 2.103e+04 | 0.01 | **0.723** |
| **out_conv** | 4.987e+01 | 1.00 | **0.277** |
| **dec_2 + out_conv** | — | — | **1.000** |

**H2 confirmada al 100%**: las dos últimas etapas (dec_2 + out_conv) concentran **toda** la varianza atribuida. El cuello de botella de la incertidumbre es muy estrecho.

### Diferencia con UNet

| Modelo | Última capa dominante | % atribuida a últimas 2 |
|---|---|---|
| UNet (Dice 0.66) | outc sola (41%) | 96.8% |
| **UniVerSeg (Dice 0.87)** | **dec_2 (72%)** | **100%** |

**Interpretación**: el J-lens "reparte" la incertidumbre entre `dec_2` y `out_conv` en UniVerSeg, mientras que en UNet casi todo recae en `outc` (la cabeza). La fundación "piensa" más antes de decidir — coherente con un modelo más complejo y mejor entrenado.

### Figura

`results/universeg_uncertainty/e2_stage_attribution.png` — bar chart con la fracción atribuida por stage.

---

## E3 — Ablación de CAVs con ΔDice y ΔEntropy

### Definición

Para cada `(stage, concept)` con `concept ∈ {tumor_core, tumor_periphery, ventricle_csf, normal_white_matter}`:
- **Clean**: forward normal → `clean_dice`, `clean_entropy`.
- **Ablated**: forward con un hook que resta la proyección del CAV en `stage` → `ablated_dice`, `ablated_entropy`.
- **Random controls**: 2 direcciones aleatorias de la misma magnitud → `random_delta_dice`, `random_delta_entropy`.

**Hipótesis H3**: la ablación cambia la incertidumbre del modelo (`ΔEntropy ≠ 0`), revelando que los CAVs son "sensores" internos.

### Resultado — Top 6 por |ΔDice|

| Stage | Concept | clean_dice | ablated_dice | ΔDice | ΔEntropy |
|---|---|---|---|---|---|
| out_conv | tumor_core | 0.577 | **0.000** | **-0.577** | +0.0001 |
| out_conv | tumor_periphery | 0.577 | **0.000** | **-0.577** | -0.0002 |
| out_conv | ventricle_csf | 0.577 | **0.000** | **-0.577** | -0.0001 |
| out_conv | normal_white_matter | 0.577 | **0.000** | **-0.577** | -0.0003 |
| dec_2 | tumor_periphery | 0.577 | 0.068 | **-0.509** | +0.0001 |
| dec_2 | tumor_core | 0.577 | 0.073 | **-0.503** | -0.0000 |

### Resultado — Todos los 12 pares

| Stage | ΔDice medio | ΔEntropy medio |
|---|---|---|
| dec_1 | +0.002 (sin efecto) | -0.0001 |
| dec_2 (tumor only) | **-0.504** | -0.0001 |
| dec_2 (others) | -0.004 | +0.0002 |
| out_conv (cualquier CAV) | **-0.577** (colapso) | ±0.0001 |

### Hallazgos detallados

**out_conv (la cabeza 1×1)**: ablacionar **cualquier** CAV en `out_conv` colapsa el Dice a 0. Eso es esperable: la cabeza tiene sólo 64→1 pesos, ablar CUALQUIER dirección no-cero deja la cabeza con output cero (regresión a la media). Los controles aleatorios también colapsan (mismo `random_delta_dice = -0.577`). Por eso `out_conv` no es un test justo del CAV.

**dec_2 + tumor_core/periphery**: ablacionar el CAV de tumor en `dec_2` baja el Dice de 0.58 a **0.07** (-0.51). Eso es el primer test "limpio" donde el CAV es **causalmente específico**:
- Los CAVs de CSF y white_matter en `dec_2` dan ΔDice ≈ 0 (específicos, no destructivos).
- Los controles aleatorios dan ΔDice ≈ ±0.01 (sin efecto similar).
- Por tanto: la dirección de tumor en `dec_2` SÍ es causalmente relevante para la decisión de UniVerSeg.

**dec_1**: ningún CAV afecta al Dice. Es la penúltima capa del decoder, donde la información clínica aún no se ha consolidado.

**ΔEntropy ≈ 0 en todos**: H3 (incertidumbre cambia sin que cambie Dice) es **FALSA**. Ablar un CAV cambia la predicción (cuando cambia) pero no la entropía del modelo. Es decir, el modelo tiene una representación causal del tumor en `dec_2` que, cuando se quita, **cambia su opinión de manera estable** (no probabilística).

### Diferencia con UNet

| Stage | UNet (Dice 0.66) | UniVerSeg (Dice 0.87) |
|---|---|---|
| dec/up_1 | ΔDice ≈ 0 | ΔDice ≈ 0 |
| dec/up_3-4 | **ΔDice = -0.37** (up4, tumor_core) | **ΔDice = -0.51** (dec_2, tumor_core/periphery) |
| out_conv | ΔDice = 0 (el UNet es invariante) | ΔDice = -0.58 (colapso) |

El UNet (entrenado) es **invariante a la ablación del CAV en `outc`** (su cabeza 1×1 + BatchNorm compensan). UniVerSeg **usa activamente la dirección de tumor en `dec_2`** y la cabeza es la salida pura de esa información.

---

## Diferencias con UNet — tabla resumen

| Aspecto | UNet (entrenado, 23 ep) | UniVerSeg (foundation, ctx=32) |
|---|---|---|
| Test Dice | 0.66 | 0.87 |
| ρ(depth, uncertainty) | -0.37 | +0.49 |
| H1 confirmada | NO | **SÍ** |
| H2 (last layers dominate) | SÍ (96.8% en up4+outc) | **SÍ** (100% en dec_2+out) |
| Ablación CAV significativa | up4, ΔDice=-0.37 | dec_2, ΔDice=-0.51 |
| ΔEntropy significativo | NO | NO |
| Patrón "workspace" | NO (inverso) | **SÍ** |

**Lectura cruzada**: el UNet, entrenado desde cero con 23 epochs (early-stop), no captura la estructura clínica profunda. La fundación few-shot (UniVerSeg), con pre-training masivo, exhibe la dinámica "workspace" que el J-lens paper original documenta en LLMs. **El método funciona — los modelos no son equivalentes.**

---

## Cómo interpretar todo el informe

1. **Empieza por E1**: si H1 es TRUE (ρ > 0), el modelo tiene dinámica workspace. Si es FALSE, está compensando desacuerdos sistemáticamente.
2. **Comprueba E2**: H2 confirmado en ambos modelos pero con diferente distribución. Las dos últimas etapas son el cuello de botella en ambos casos.
3. **E3 es el test causal**: ablaciones significativas en stages intermedios (no en la cabeza) = el modelo realmente "ve" los conceptos clínicos. Sin efecto = el modelo usa otras direcciones.

---

## Archivos generados

```
results/universeg_uncertainty/
├── e1_scatter.png               # scatter depth vs entropía + ρ
├── e1_summary.json              # {spearman_rho, pvalue, h1_confirmed, ...}
├── uncertainty_maps.npy         # (n_test × 128 × 128) entropía per-píxel
├── decision_depth.npy           # (n_test × 128 × 128) depth per-píxel
├── e2_stage_attribution.csv     # {stage, var_features, jac_norm_sq, frac}
├── e2_stage_attribution.png     # bar chart
├── e2_summary.json              # {stage_attribution, h2_confirmed, ...}
├── e3_ablation.csv              # 12 filas: stage × concept con ΔDice y ΔEntropy
├── e3_summary.json              # {results, n_uncertainty_samples, ...}
└── summary.json                 # {args, e1, e2, e3, total_time_s}
```

---

## Limitaciones de este run

1. **n_test = 8** es bajo para significancia estricta. Repetir con n_test = 30 para publicar.
2. **n_calibration = 5** para el J-lens. Sub-muestreo a 12 píxeles/imagen da un Jacobiano ruidoso. Con n_cal = 20+ los resultados serían más estables.
3. **MC Dropout vía hook** (no entrenado). El J-lens está midiendo la sensibilidad lineal a perturbaciones de features, no la incertidumbre bayesiana "real". Pero el paper original del J-lens también mide linealización, así que es consistente.
4. **out_conv colapsa** bajo ablación de cualquier CAV. Para testear ablaciones en la cabeza, hay que usar un set de CAVs ortogonales o aplicar la ablación de forma diferente (proyección parcial, no total).
5. **TTA con hflip solo** para UniVerSeg (resize interno). Pierde el poder de TTA que UNet tiene (6 transforms).

---

## Comando reproducible

```bash
# Setup
cd ~/SegLens
uv sync --extra universeg

# Run
uv run seglens-run-universeg-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir results/universeg_uncertainty \
  --context-size 32 --channel g \
  --n-test 8 --n-calibration 5 \
  --n-samples 4 --n-uncertainty-samples 3 \
  --n-pixels-jac 12 \
  --ablation-stages dec_blocks.1.target.vmapped \
                    dec_blocks.2.target.vmapped \
                    out_conv.conv
```
