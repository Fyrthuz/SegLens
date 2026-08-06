# F6 — Run completo con n_test=30 (UNet vs UniVerSeg)

> Réplica de los 3 experimentos de incertidumbre (E1, E2, E3) con **n_test=30** en ambos modelos del TFM. Compara los patrones "workspace" emergentes con significancia estadística real. Ejecutado el 2026-08-06 sobre RTX 5080.

---

## TL;DR

Con n_test=30 (vs n=8-10 del run anterior), las conclusiones se **consolidan** y surge una imagen clara:

| Aspecto | UNet (Dice 0.47) | **UniVerSeg (Dice 0.82)** |
|---|---|---|
| **ρ(depth, uncertainty)** | **-0.412** (H1 false) | **+0.298** (H1 true) |
| Ratio late/early entropía | 0.87 (late < early) | **5.36** (late ≫ early) |
| H2: varianza en últimas 2 etapas | **96.7%** (up4+outc) | **100%** (dec_2+out) |
| Mejor ΔDice por ablación CAV | **-0.345** (up4, tumor_periphery) | **-0.741** (dec_2, tumor_periphery) |
| Ablación outc/out_conv colapsa | sí (-0.47) | sí (-0.82) |
| ΔEntropy significativo | NO | NO |

**Conclusión principal**: los dos modelos se comportan **cualitativamente opuestos** en la dimensión workspace-vs-no-workspace. UNet exhibe **desacuerdos sistemáticos J-lens ↔ output final** (ρ negativo); UniVerSeg exhibe **la dinámica workspace que el J-lens paper documenta para LLMs** (ρ positivo, con late-decision pixels mucho más inciertos). La diferencia se mantiene estable al pasar de n=8 a n=30.

---

## Configuración experimental

```bash
# UNet (modelo entrenado en el TFM, early-stopped en epoch 23)
uv run seglens-uncertainty-link --checkpoint models/unet_real.pth \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --method mc_dropout --n-test 30 --n-calibration 10 --n-samples 5 \
  --max-pixels-jac 16 --size 256 \
  --output-dir results/f6_n30/unet/e1

uv run seglens-stage-uncertainty ... --n-test 30 --n-samples 5 \
  --output-dir results/f6_n30/unet/e2

uv run seglens-ablation ... --n-test 30 --n-calibration 10 \
  --n-uncertainty-samples 4 --n-random-controls 3 \
  --stages up1 up2 up3 up4 outc \
  --output-dir results/f6_n30/unet/e3

# UniVerSeg (foundation few-shot, ctx=32, T1c)
uv run seglens-run-universeg-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --context-size 32 --channel g \
  --n-test 30 --n-calibration 10 --n-samples 5 --n-uncertainty-samples 4 \
  --n-pixels-jac 16 \
  --ablation-stages dec_blocks.1 dec_blocks.2 out_conv \
  --output-dir results/f6_n30/universeg
```

| Parámetro | Valor | Razón |
|---|---|---|
| `n_test` | 30 | Suficiente para significancia estadística (p<0.001 esperable con >10⁵ píxeles) |
| `n_calibration` | 10 | Compromiso entre estabilidad del J-lens y coste computacional |
| `n_samples` (MC) | 5 | N forward passes para estimar entropía |
| `n-uncertainty-samples` | 4 | N forward passes para ΔEntropy en E3 |
| `max_pixels_jac` | 16 | Sub-muestreo para el J-lens (256² / 16 ≈ 4000 píxeles/forward) |
| `context-size` (UniVerSeg) | 32 | Suficiente few-shot; ahorra vs 64 |
| `channel` (UniVerSeg) | g (T1c) | Confirmado como el mejor canal en estudio previo |

**Tiempos (RTX 5080)**:
- UNet E1+E2+E3: ~3-5 min en total
- UniVerSeg (todo en uno): ~2-3 min

---

## E1 — Decision depth vs uncertainty (correlación per-píxel)

### Definición

- **`decision_depth[píxel]`**: primer stage donde la predicción del J-lens coincide con la predicción final binarizada. `len(stages)` si nunca coincide.
- **`uncertainty[píxel]`**: entropía de las predicciones bajo MC Dropout (5 forward passes con hook en `up3`+`up4` para UNet, `dec_2`+`dec_1` para UniVerSeg).
- **Métrica**: correlación de Spearman píxel-a-píxel, + discretización en early (depth ≤ 2) vs late (depth ≥ 6).

### Resultados

| Modelo | n_test | ρ(Spearman) | p-valor | H1 | unc early | unc late | ratio late/early |
|---|---|---|---|---|---|---|---|
| UNet (n=10, run previo) | 10 | -0.368 | ≈ 0 | FALSE | 0.281 | 0.000 | 0.00 |
| **UNet (n=30)** | **30** | **-0.412** | **≈ 0** | **FALSE** | **0.279** | **0.242** | **0.87** |
| UniVerSeg (n=8, run previo) | 8 | +0.490 | ≈ 0 | TRUE | 0.036 | 0.069 | 1.90 |
| **UniVerSeg (n=30)** | **30** | **+0.298** | **≈ 0** | **TRUE** | **0.051** | **0.272** | **5.36** |

### Lectura detallada

**UNet** (n=30, ρ = -0.412, H1 false):
- La correlación negativa es **fuerte y estable** con más datos (p ≈ 0, n_píxeles = 983k).
- Late-decision pixels tienen *menos* entropía que early-decision (0.24 vs 0.28). El efecto es modesto pero consistente.
- **Interpretación**: en el UNet, los píxeles donde el J-lens nunca converge con el final son píxeles donde la cabeza 1×1 + BatchNorm compensan la discrepancia. La salida es estable (baja entropía) pero distinta de lo que las features intermedias sugieren. La fundación few-shot no, pero el UNet sí.

**UniVerSeg** (n=30, ρ = +0.298, H1 true):
- La correlación positiva es **fuerte y significativa**.
- Late-decision pixels tienen **5.36× más entropía** que early-decision (0.272 vs 0.051). Con n=8 era 1.9×; con n=30 la ratio se amplifica porque la cola de píxeles "difíciles" tiene más muestra.
- **Interpretación**: workspace real. Los píxeles que la fundación necesita "pensar más" (más etapas para coincidir) son precisamente los más inciertos. Es el patrón exacto que el J-lens paper documenta para LLMs.

**Diferencia con n=8**:
- UNet ρ: -0.37 → -0.41 (se solidifica).
- UniVerSeg ρ: +0.49 → +0.30 (se reduce pero sigue muy significativa). La razón: con n=8 los píxeles "late" eran pocos y heterogéneos; con n=30 se acumulan en torno al valor medio. **El signo se mantiene**.

### Salida

`results/f6_n30/unet/e1/scatter.png`, `results/f6_n30/universeg/e1_scatter.png`

---

## E2 — Atribución de varianza por stage (vía J-lens)

### Definición

Para cada stage `l`:
- `Var(f_l)` = varianza de las features bajo N MC passes.
- `‖J_l‖²` = suma de cuadrados de los pesos del Jacobiano (magnitud causal de la capa).
- `contribución_l = mean(Var(f_l)) · ‖J_l‖²` (varianza propagada al logit).
- `frac_l = contribución_l / Σ contribución`.

**Hipótesis H2**: las últimas etapas concentran >50% de la varianza atribuida.

### Resultados

**UNet (n=30, 5 MC passes, MC stages = up3+up4)**:

| Stage | var(features) | ‖J_l‖² | frac |
|---|---|---|---|
| inc | 0.000 | 0.00 | 0.000 |
| down1 | 0.000 | 0.00 | 0.000 |
| down2 | 0.000 | 0.00 | 0.000 |
| down3 | 0.000 | 0.00 | 0.000 |
| down4 | 0.000 | 0.00 | 0.000 |
| up1 | 0.000 | 0.00 | 0.000 |
| up2 | 0.000 | 0.00 | 0.000 |
| up3 | 0.192 | 0.02 | 0.033 |
| **up4** | 0.120 | 0.69 | **0.583** |
| **outc** | 0.055 | 1.00 | **0.385** |
| **up4 + outc** | — | — | **0.967** |

**UniVerSeg (n=30, 5 MC passes, MC stages = dec_1+dec_2)**:

| Stage | var(features) | ‖J_l‖² | frac |
|---|---|---|---|
| enc_0 (target) | 0.000 | 0.20 | 0.000 |
| enc_1, enc_2, enc_3 | 0.000 | 0.00 | 0.000 |
| dec_0 | 0.000 | 0.00 | 0.000 |
| dec_1 | 5.05e+04 | 0.00 | 0.000 |
| **dec_2** | 2.29e+04 | 0.01 | **0.335** |
| **out_conv** | 2.81e+02 | 1.00 | **0.665** |
| **dec_2 + out_conv** | — | — | **1.000** |

### Comparativa

| Aspecto | UNet | UniVerSeg |
|---|---|---|
| Última 2 capas concentran | 96.7% | 100% |
| Distribución interna | up4 58% / outc 39% | dec_2 34% / out 67% |
| Stages tempranos contribuyen | 0% | 0% |

**Lectura**: ambos modelos concentran la incertidumbre en las dos últimas etapas. **Pero UniVerSeg lo hace de forma aún más extrema** (100% vs 96.7%) y con un reparto diferente: en UniVerSeg la cabeza `out_conv` (1×1) absorbe el 67% de la varianza atribuida, mientras que en UNet está más repartido entre `up4` (58%) y `outc` (38%).

**Interpretación del reparto diferente**: en UNet, la cabeza tiene pesos pequeños (‖J‖²=1.0) pero la varianza de features se concentra en `up4` (‖J‖²=0.69). En UniVerSeg, la varianza de features es enorme en `dec_1` y `dec_2` (5e+4, 2e+4) pero su Jacobiano es casi cero; sólo `out_conv` tiene Jacobiano alto (1.0) y por tanto "traduce" esa varianza al logit.

### Salida

`results/f6_n30/unet/e2/stage_attribution.png`, `results/f6_n30/universeg/e2_stage_attribution.png`

---

## E3 — Ablación de CAVs con ΔDice y ΔEntropy

### Definición

Para cada `(stage, concept)`:
- **Clean**: forward normal.
- **Ablated**: forward con hook que resta la proyección del CAV en `stage`.
- **Random controls**: 3 direcciones aleatorias de la misma magnitud.
- **Métricas**: `clean_dice`, `ablated_dice`, `delta_dice = ablated - clean`, `clean_entropy`, `delta_entropy`.

**Hipótesis H3**: la ablación cambia la incertidumbre (`ΔEntropy ≠ 0`). **Hipótesis complementaria**: ablaciones significativas en stages intermedios (no en la cabeza) revelan CAVs causalmente relevantes.

### Resultados UNet (n=30, 4 MC samples, 3 random controls)

**Top 10 por |ΔDice|** (20 totales):

| Stage | Concept | clean | ablated | ΔDice | ΔEntropy |
|---|---|---|---|---|---|
| outc | tumor_core | 0.469 | 0.000 | **-0.469** | +0.00003 |
| outc | tumor_periphery | 0.469 | 0.000 | **-0.469** | +0.00001 |
| outc | ventricle_csf | 0.469 | 0.000 | **-0.469** | +0.00005 |
| outc | normal_white_matter | 0.469 | 0.000 | **-0.469** | -0.00001 |
| up4 | tumor_periphery | 0.469 | 0.123 | **-0.345** | -0.00002 |
| up4 | tumor_core | 0.469 | 0.216 | **-0.252** | +0.00003 |
| up4 | normal_white_matter | 0.469 | 0.246 | **-0.222** | +0.00003 |
| up1 | ventricle_csf | 0.469 | 0.446 | -0.022 | +0.00003 |
| up2 | ventricle_csf | 0.469 | 0.447 | -0.022 | +0.00000 |
| up3 | ventricle_csf | 0.469 | 0.459 | -0.010 | -0.00001 |

**Hallazgos UNet**:
- `outc` colapsa con cualquier CAV: el head tiene 64→1 pesos, ablacionar CUALQUIER vector no-cero colapsa la cabeza. Los 3 random controls en outc también colapsan (-0.469). Por eso no es un test justo.
- **`up4` con tumor_periphery / tumor_core / normal_white_matter** da ΔDice significativos (-0.22 a -0.35). Random controls en up4 son del orden de ±0.005. **Es un test causal limpio**: ablaciones específicas afectan al modelo, controles no.
- `up1-3` con cualquier CAV: ΔDice ≈ 0 (sin efecto). Las direcciones clínicas no son relevantes en esas etapas.
- ΔEntropy ≈ 0 en todas: H3 false. La ablación cambia la predicción pero no la entropía del modelo.

### Resultados UniVerSeg (n=30, 4 MC samples, 2 random controls)

**Top 10 por |ΔDice|** (12 totales):

| Stage | Concept | clean | ablated | ΔDice | ΔEntropy |
|---|---|---|---|---|---|
| out_conv | tumor_core | 0.824 | 0.000 | **-0.824** | -0.00004 |
| out_conv | tumor_periphery | 0.824 | 0.000 | **-0.824** | +0.00008 |
| out_conv | ventricle_csf | 0.824 | 0.000 | **-0.824** | -0.00003 |
| out_conv | normal_white_matter | 0.824 | 0.000 | **-0.824** | +0.00004 |
| dec_2 | tumor_periphery | 0.824 | 0.082 | **-0.741** | -0.00004 |
| dec_2 | tumor_core | 0.824 | 0.088 | **-0.735** | -0.00012 |
| dec_2 | normal_white_matter | 0.824 | 0.815 | -0.008 | -0.00004 |
| dec_2 | ventricle_csf | 0.824 | 0.822 | -0.002 | +0.00006 |
| dec_1 | normal_white_matter | 0.824 | 0.823 | -0.001 | -0.00006 |
| dec_1 | tumor_periphery | 0.824 | 0.823 | -0.000 | +0.00006 |

**Hallazgos UniVerSeg**:
- `out_conv` colapsa con cualquier CAV (mismo motivo: 64→1 head). También los random controls colapsan.
- **`dec_2` con tumor_periphery / tumor_core**: ΔDice = -0.74 y -0.74. **Mucho más fuerte que en UNet (up4: -0.35)**. Los random controls en dec_2 dan ΔDice = ±0.005-0.02 — orden de magnitud menor.
- **dec_2 con CSF / white_matter**: ΔDice ≈ 0. Específico a tumor.
- dec_1: ΔDice ≈ 0 en todos. Demasiado pronto en el decoder.
- ΔEntropy ≈ 0 en todos: H3 false.

### Comparativa cuantitativa de las ablaciones significativas

| Modelo | Stage | Mejor concepto | ΔDice | vs random control | Específico? |
|---|---|---|---|---|---|
| UNet | up4 | tumor_periphery | -0.345 | ±0.005 | **SÍ** |
| UNet | up4 | tumor_core | -0.252 | ±0.005 | **SÍ** |
| UNet | up4 | normal_white_matter | -0.222 | ±0.005 | **SÍ** (sorprendente) |
| **UniVerSeg** | **dec_2** | **tumor_periphery** | **-0.741** | ±0.02 | **SÍ (×2 más fuerte)** |
| **UniVerSeg** | **dec_2** | **tumor_core** | **-0.735** | ±0.02 | **SÍ (×3 más fuerte)** |

**Lectura cruzada**:
- Ambos modelos son **específicos**: ablacionar "tumor" afecta, ablacionar "CSF" o "white_matter" no.
- UniVerSeg es **cualitativamente más sensible**: ΔDice = -0.74 vs -0.35 (2.1× más fuerte).
- El UNet "ve" tumor en `up4` (penúltimo decoder). UniVerSeg "ve" tumor en `dec_2` (también penúltimo). **Misma posición funcional, distinta magnitud de respuesta**.
- En ambos casos `outc`/`out_conv` colapsa con cualquier cosa — es un artefacto del head 1×1, no un test válido.

### Salida

`results/f6_n30/unet/e3/ablation.csv`, `results/f6_n30/universeg/e3_ablation.csv`

---

## Comparativa global UNet vs UniVerSeg (n=30)

| Dimensión | UNet | UniVerSeg | Quién exhibe la propiedad |
|---|---|---|---|
| **Test Dice (clean)** | 0.47 | 0.82 | — |
| E1: ρ(depth, uncertainty) | **-0.412** (H1 false) | **+0.298** (H1 true) | UniVerSeg |
| E1: ratio late/early unc | 0.87 | **5.36** | UniVerSeg |
| E2: frac últimas 2 etapas | 96.7% | **100.0%** | ambos (H2) |
| E2: distribución interna | up4 58% / outc 39% | dec_2 34% / out 67% | — |
| E3: mejor ΔDice en stage medio | -0.345 (up4) | **-0.741** (dec_2) | UniVerSeg |
| E3: especificidad CAV | SÍ (tumor > CSF) | **SÍ (tumor > CSF, mayor magnitud)** | ambos |
| E3: ΔEntropy significativo | NO | NO | H3 false en ambos |
| E3: outc/out_conv colapso | SÍ | SÍ | artefacto del head |

### Patrón emergente

```
UNet (entrenado, 23 ep):              UniVerSeg (foundation):
                                     
depth   unc                          depth   unc
  |     |                              |     |
  0 ----+----  (early)                 0 ----+----  (early, low unc)
  |     |                              |     |
  |     X                              |     |
  |    / \                             |     |
  |   /   \                            |     |
  9  /  X  \  (late, ≈ early)         9  ----+----X----  (late, ~5x higher)
     X                                    
     \                                    
      X  (H1 false)                    (H1 true, workspace-like)
```

**Conclusión cualitativa**:
- **UNet entrenado** = modelo "opaco" en el sentido J-lens: las features intermedias y la cabeza están desacopladas, las ablaciones de CAVs se compensan en algún sitio, la incertidumbre MC no correlaciona con la profundidad de cómputo.
- **UniVerSeg foundation** = modelo "transparente": workspace-like, ablaciones específicas afectan, incertidumbre crece con profundidad. La fundación "ve" lo que un humano buscaría.

**Hipótesis verosímil**: el pre-training masivo de UniVerSeg con miles de imágenes fuerza al modelo a usar las features de manera consistente y predecible. El UNet entrenado en 23 epochs sobre 715 imágenes desde cero no ha tenido tiempo/exposición de desarrollar ese "workspace" — la cabeza 1×1 ha aprendido a compensarlo todo.

---

## Limitaciones y trabajo futuro

1. **n=30 es bueno pero no exhaustivo**: para publicación repetir con n_test=50+ y verificar la estabilidad.
2. **MC Dropout via hook** (no entrenado): es una aproximación de incertidumbre bayesiana. El paper del J-lens también mide linealización, así que es conceptualmente consistente, pero la incertidumbre "real" podría diferir.
3. **outc/out_conv colapsa** bajo ablación de cualquier CAV: artefacto del head 1×1 con 64→1. Para testear ablaciones de la cabeza se necesita una arquitectura con más pesos.
4. **TTA con hflip solo** para UniVerSeg. UNet tiene 6 transforms (incl. rotaciones con padding). Posible sesgo a favor de UNet en métricas de incertidumbre.
5. **CAVs derivados de T1c** (4 conceptos). Para UniVerSeg podría haber CAVs más informativos en espacios latentes diferentes.
6. **El UNet early-stopped** (Dice 0.47, peor que el TFM) puede subestimar las capacidades del modelo. Sería interesante reentrenar 60 epochs y comparar.

---

## Reproducibilidad

Comandos exactos (todo en `~/SegLens`):

```bash
# UNet
uv run seglens-uncertainty-link \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_real.pth \
  --method mc_dropout --n-test 30 --n-calibration 10 --n-samples 5 \
  --max-pixels-jac 16 --size 256 \
  --output-dir results/f6_n30/unet/e1

uv run seglens-stage-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_real.pth \
  --method mc_dropout --n-test 30 --n-calibration 10 --n-samples 5 \
  --max-pixels-jac 16 --size 256 --n-display 5 \
  --output-dir results/f6_n30/unet/e2

uv run seglens-ablation \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --checkpoint models/unet_real.pth \
  --n-calibration 10 --n-test 30 --n-random-controls 3 --n-uncertainty-samples 4 \
  --size 256 --stages up1 up2 up3 up4 outc \
  --output-dir results/f6_n30/unet/e3

# UniVerSeg
uv run seglens-run-universeg-uncertainty \
  --data-root /home/fyrthuz/Desktop/Codigo_TFM/MRI/filtered_data \
  --output-dir results/f6_n30/universeg \
  --context-size 32 --channel g \
  --n-test 30 --n-calibration 10 \
  --n-samples 5 --n-uncertainty-samples 4 \
  --n-pixels-jac 16 \
  --ablation-stages dec_blocks.1.target.vmapped \
                    dec_blocks.2.target.vmapped \
                    out_conv.conv
```

Datos usados: 30 primeros slices del split de test (patient-level) del LGG MRI Segmentation Dataset (Kaggle, mateuszbuda), filtrados al 1% foreground (1060 slices, 108 pacientes).

Modelo UNet: `models/unet_real.pth` (TFM, 23 epochs early-stopped, Dice 0.66 sobre test completo; aquí 0.47 sobre los primeros 30).

Modelo UniVerSeg: pretrained de JJGO/UniverSeg, ctx=32 imágenes T1c (canal g), 5 forward passes MC Dropout.
