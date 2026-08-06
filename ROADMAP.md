# ROADMAP

> Plan de implementación de SegLens. Adaptación de la mecánica del J-lens (Anthropic, 2026) a segmentación de imagen médica, agnóstica del modelo.

---

## 0. Punto de partida

**Lo que se traslada del paper original** (mecánica):
- Cálculo de un Jacobiano promediado por capa `E_x[∂y/∂f_l]` como mapa lineal causal de etapa → salida.
- Comparativa de tres lentes con naturalezas distintas (naive correlacional / probe correlacional / causal Jacobiano).
- Intervenciones sobre direcciones interpretables (ablación, steering, patching).
- Conceptos como direcciones indexadas en espacio de features (sustituyendo "tokens" por "conceptos clínicos").

**Lo que NO se traslada** (exclusivo de LLMs):
- Vocabulario discreto y unembedding textual.
- Reporte verbal, modulación por instrucciones, introspección.
- Narrativa de "global workspace" / consciencia funcional.
- Counterfactual reflection training.

**Análogos formales** (lo que se hace en segmentación):

| J-lens (LLM) | SegLens (segmentación) |
|---|---|
| Residual stream de ancho constante | Feature map de etapa (cambia de canales y resolución) |
| Token embedding + positional | Píxel + coordenadas espaciales |
| `W_U` (unembedding) | Cabeza 1×1 final |
| Vocabulario de tokens | Conjunto de conceptos clínicos (T1c) |
| Top-1 token = "lo que el modelo va a decir" | Mapa de segmentación o score de concepto = "lo que la etapa ya decide" |
| J-lens vector `v_t` | Concept activation vector (CAV) por concepto |
| J-space (combinación sparse de v_t) | Score por píxel sobre el conjunto de CAVs |

---

## 1. Fases de implementación

### F0 — Adapter protocol ✅ *(prioridad alta)*

Objetivo: definir el contrato mínimo que cualquier modelo de segmentación debe cumplir para ser analizable.

**Salidas verificables**:
- `SegmentationAdapter` (Protocol) con `name`, `stages`, `segment()`, `head()`, `features()`, `preprocess()`.
- `UNetAdapter` funcional (hooks triviales sobre `inc`/`down{1..4}`/`up{1..4}`/`outc`).
- `UniVerSegAdapter` funcional (introspección del `universeg` interno via `named_modules()`).
- `tests/test_adapter.py` verde.

### F1 — Núcleo de las 3 lentes ✅ *(prioridad alta)*

Objetivo: implementar los tres mecanismos de lectura por etapa.

- `FeatureExtractor`: context manager con hooks por nombre, devuelve `dict[str, Tensor]`.
- **Logit lens** (`lens/logit_lens.py`): aplica `head()` a las features (con upsampling si la resolución no coincide). En UNet solo aplicable a `up4` (mismo nº de canales que la cabeza); se documenta la limitación.
- **Tuned lens** (`lens/tuned_lens.py`): probe lineal por etapa `W_l ∈ R^{C_out × C_l}` entrenada minimizando `||W_l f_l - y||²` sobre N imágenes de calibración. Bilinear upsampling de `f_l` a la resolución de `y` antes del probe.
- **J-lens** (`lens/jacobian_lens.py`): Jacobiano por píxel promediado. Para cada imagen `x` con `B·H·W` píxeles y feature de etapa `f_l ∈ R^{C_l}` por píxel, calcula `∂y_p / ∂f_{l,p}` (bloque diagonal por píxel), promedia sobre imágenes y posiciones. Aplica `J_l · f_l` por píxel → mapa de logits de la misma forma que la cabeza.

Tests unitarios: probes convergen a alta accuracy con modelo trivial; J-lens reproduce la cabeza en la última capa; magnitudes razonables.

### F2 — Curva de profundidad de decisión ⏳

Objetivo: ejecutar el experimento E1–E3 sobre UNet y UniVerSeg con un dataset pequeño (subset de LGG).

- `experiments/run_calibration.py` (E1): ajusta probes y Jacobianos sobre N=200 imágenes de validación.
- `experiments/run_depth_curves.py` (E2): para cada imagen de test, lee la segmentación con cada lente en cada etapa; reporta Dice/IoU medio. Produce figura: Dice-vs-etapa, 3 curvas por modelo.
- `experiments/run_qualitative.py` (E3): para 10 imágenes, muestra mapas por lente a 3 profundidades seleccionadas.

**Entregable**: `results/depth_curves.png` + tabla CSV.

### F3 — Conceptos clínicos desde T1c ⏳

Objetivo: análogo al "vocabulario de tokens" del paper. Sin BraTS: derivar conceptos a partir de la máscara binaria y la intensidad T1c.

- `concepts.py`:
  - `ConceptThresholds` (percentiles T1c, sobre Foreground / Background).
  - `derive_concept_masks(image, mask, t1c_channel=1)` → `dict[name, Tensor]`.
  - `CAV` (concept activation vector) por etapa: `v_{l,c} = E[f_l | mask_c] - E[f_l | ¬mask_c]`.
  - `ConceptScorer`: por píxel, `score_l(x, c) = <f_l(x), v_{l,c}>` → heatmap.

**Conceptos definidos**:

| Concepto | Definición operativa |
|---|---|
| `tumor_core` | Foreground ∩ T1c > p90(FG) |
| `tumor_periphery` | Foreground ∩ p60 ≤ T1c ≤ p90 (FG) |
| `ventricle_csf` | Background ∩ T1c < p15 (BG) |
| `normal_white_matter` | Background ∩ p40 ≤ T1c ≤ p70 (BG) |

**Entregable**: `experiments/run_concept_scoring.py` + heatmaps cualitativos.

### F4 — Intervenciones ⏳

Objetivo: claims causales (no solo correlacionales).

- `interventions.py`:
  - `ablate_direction(f, v)`: `f ← f - <f, v̂> · v̂` (proyecta fuera).
  - `steer(f, v, α)`: `f ← f + α·v`.
  - `patch_activations(f_src, f_tgt, mask)`: copia activaciones de `f_tgt` en `f_src` según `mask` espacial.
- `experiments/run_ablation.py` (E5): ablaciona CAVs por etapa, mide ΔDice vs dirección aleatoria de magnitud equiparable (mismo ‖v‖, dirección sampleada de la esfera). Tabla: ΔDice × etapa × concepto.

**Entregable**: tabla CSV de ablaciones, plot tipo "ablation heatmap".

### F5 — Comparativa cruzada UNet vs UniVerSeg ⏳

Objetivo: el resultado más limpio y publicable del proyecto.

- `experiments/run_model_comparison.py` (E6): re-ejecuta F2 y F4 sobre ambos modelos con la misma calibración. Genera:
  - Figura doble: curvas de profundidad lado a lado.
  - Tabla cruzada: ¿dónde "decide" cada modelo? ¿qué conceptos usa más?
  - Discusión: ¿la ventaja de UNet (Dice 0.89 vs 0.76 en test LGG) viene de "decidir antes" o de "decidir mejor"?

**Entregable**: `results/model_comparison.{png,csv,md}`.

### F6 — Integración con incertidumbre ⏳

Objetivo: cerrar el círculo con la línea principal del TFM original.

- Cargar mapas de incertidumbre existentes (MC Dropout, TTA, Fusion) del pipeline del TFM.
- Para cada píxel, registrar: (a) la etapa en la que el J-lens alcanza la decisión final (curva E2), (b) la incertidumbre.
- Plot: dispersión "etapa de decisión vs incertidumbre" + correlaciones.
- Hipótesis a verificar: píxeles de alta incertidumbre son píxeles de "decisión tardía".

**Entregable**: `results/uncertainty_vs_depth.png`.

### F7 — Visualización, tests finales, redacción ⏳

- `viz.py`: helpers para figuras finales (estilo TFM).
- Cobertura de tests > 80% en `seglens/`.
- `docs/METHODOLOGY.md`: descripción matemática rigurosa de las 3 lentes y su relación con el paper.
- `docs/RESULTS.md`: tablas y figuras finales.
- `examples/notebook_demo.ipynb`: walkthrough ejecutable end-to-end sobre 10 imágenes de LGG.

---

## 2. Estimación de esfuerzo

| Fase | Horas |
|---|---|
| F0 | 4–6 |
| F1 | 8–12 |
| F2 | 4–6 |
| F3 | 6–8 |
| F4 | 4–6 |
| F5 | 6–10 |
| F6 | 3–4 |
| F7 | 6–8 |
| **Total** | **~40–60 h** |

Apropiado para un capítulo de TFM o un side-project serio.

---

## 3. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `universeg` no expone hooks limpios | F0 lo verifica primero; fallback a lentes input-output only + atribución (IG) |
| Jacobianos ruidosos en CNN con BatchNorm | Forzar `eval()`; congelar `head` durante calibración; comparar `torch.autograd.functional.jacobian` con muestreo manual |
| Conceptos T1c mal calibrados | Análisis de histogramas previo; reportar sensibilidad a percentiles ±5 |
| Curvas de profundidad planas o ruidosas | Baselines: dirección aleatoria, "shuffle de píxeles" como control inferior |
| Memoria de Jacobianos (UNet bottleneck: 1024 × 128² píxeles) | Procesar por mini-batches de píxeles; backprop por imagen; ~32 imágenes de calibración bastan |

---

## 4. Lo que NO se hace (importante)

- No se replica la narrativa de "workspace verbalizable" — no aplica a segmentación.
- No se intenta introspección ni modulación por instrucciones.
- No se reentrena ningún modelo.
- No se compara con el paper como si fuera una réplica; se cita como **adaptación metodológica**.

---

## 5. Criterios de éxito

El proyecto se considera completo cuando:

1. ✅ `pytest tests/ -v` pasa al 100%.
2. ✅ `experiments/run_depth_curves.py` produce la figura de 3 lentes × 2 modelos sobre el dataset LGG.
3. ✅ `experiments/run_ablation.py` produce la tabla de ablaciones con baselines.
4. ✅ Existe un notebook ejecutable end-to-end (`examples/notebook_demo.ipynb`).
5. ✅ La documentación (`docs/METHODOLOGY.md`) deja explícitas las diferencias con el paper original.
