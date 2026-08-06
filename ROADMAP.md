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

### F0 — Adapter protocol ✅

### F1 — Núcleo de las 3 lentes ✅

### F2 — Curva de profundidad de decisión ✅

### F3 — Conceptos clínicos desde T1c ✅

### F4 — Intervenciones ✅

### F5 — Comparativa cruzada UNet vs UniVerSeg ✅

### F6 — Integración con incertidumbre ✅

**Componentes**:
- `seglens/uncertainty.py`: `MCDropoutInjector` (hook-based, sin reentrenar) + `UncertaintyEstimator` con `mc_dropout` / `tta` / `noisy`.
- `adapter.uncertainty(x, method, n_samples)`: API unificada.
- **E1 `seglens-uncertainty-link`**: correlación per-píxel decision-depth vs uncertainty.
- **E2 `seglens-stage-uncertainty`**: atribución de varianza por stage vía J-lens (`var(f_l) · ‖J_l‖²`).
- **E3 extensión de `seglens-ablation`**: ΔDice + ΔEntropy con `--n-uncertainty-samples`.
- `seglens-depth-curves --with-uncertainty`: bandas sombreadas en la figura principal.
- 17 tests nuevos (`tests/test_uncertainty.py`).

**Hipótesis validadas**:
- **H1** `ρ(depth, uncertainty) > 0`: FALSA — correlación **negativa** en nuestro UNet (píxeles "decididos tarde" tienen **menos** incertidumbre, son desacuerdos sistemáticos J-lens vs final).
- **H2** últimas etapas concentran >50% de la incertidumbre: **TRUE** — up4+outc concentran **96.8%** de la varianza atribuida.
- **H3** `ΔEntropy ≠ 0` con `ΔDice ≈ 0`: FALSA — pero surge **nuevo hallazgo**: la ablación SÍ funciona en `up4` con ΔDice = **-0.37** para `tumor_core` (vs 0 en `up3`/`outc`).

### F7 — Visualización, tests finales, redacción ⏳

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
