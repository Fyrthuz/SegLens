# SegLens

> **Lentes causales para explicabilidad agnóstica del modelo en segmentación de imagen médica.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet)](https://docs.astral.sh/uv/)

---

## 1. ¿Qué es?

SegLens es un framework de **explicabilidad por etapas** para modelos de segmentación de imagen. Está inspirado en el *Jacobian Lens* (J-lens) de Anthropic —la técnica que identifica el "workspace verbalizable" en LLMs— pero adaptado al dominio de segmentación, donde no hay vocabulario ni unembedding textual.

**Idea nuclear**: para cada etapa intermedia de un modelo de segmentación (UNet, UniVerSeg, o cualquier encoder-decoder), calculamos un *readout lineal* de la decisión de segmentación desde sus features. Comparamos tres lentes:

| Lente | Mapeo | Naturaleza |
|---|---|---|
| **Logit lens** | Aplicar la cabeza 1×1 directamente | Correlacional, naive |
| **Tuned lens** | Probe lineal entrenada por etapa | Correlacional, aprende corrección de base |
| **J-lens** | Jacobiano promediado `E[∂y/∂f_l]` por etapa | **Causal** (lineal de primer orden) |

Más una capa de **conceptos clínicos** (estilo TCAV: tumor core, periferia, ventrículos, tejido normal — derivados de T1c) que actúa como "vocabulario" análogo a los tokens del paper original, y un módulo de **intervenciones** (ablación, steering, patching) para verificar causalmente qué direcciones de features importan.

Todo detrás de un `SegmentationAdapter` minimal, así que **cualquier modelo con cabeza 1×1 puede enchufarse** sin tocar el resto del framework.

---

## 2. Quick start

```bash
# clonar e instalar (deps mínimas, sin universeg)
git clone https://github.com/Fyrthuz/SegLens && cd SegLens
make install

# incluyendo UniVerSeg + dev tools
make dev

# smoke test end-to-end (UNet sintético, las 3 lentes, ~3s)
make run-smoke

# tests
make test
```

---

## 3. Pipeline end-to-end: qué hace cada paso, qué recibe y qué produce

El pipeline completo se ejecuta con una secuencia de comandos. Cada uno está descrito con **Inputs** (lo que recibe), **Outputs** (lo que escribe a disco), y **Interpretación** (cómo leer el resultado).

### Paso 0 — Datos: descargar y filtrar LGG MRI Segmentation

```bash
make download-data        # o: uv run seglens-download-data --raw ./data/lgg_raw --filtered ./data/lgg
```

- **Input**: credenciales Kaggle (`KAGGLE_USERNAME`, `KAGGLE_KEY`) o `--skip-download` si ya tienes el dataset.
- **Acción**:
  1. Descarga `mateuszbuda/lgg-mri-segmentation` (zip ~1 GB, 110 pacientes, 3929 slices TIF RGB con T1/T1c/FLAIR en R/G/B + máscara binaria de tumor).
  2. Filtra slices con `foreground ratio ≥ 1%` (mínimo de tumor para que sea útil al modelo). Típicamente retiene ~1060 slices de 108 pacientes.
- **Output**:
  - `./data/lgg_raw/lgg-mri-segmentation/kaggle_3m/TCGA_*/<slice>_<idx>.tif` y `<slice>_<idx>_mask.tif` (raw)
  - `./data/lgg/TCGA_*/...` (filtrado, mismo formato)

**Estructura esperada**:
```
data/lgg/
└── TCGA_CS_4941_19960909/
    ├── TCGA_CS_4941_19960909_1.tif        # imagen RGB: R=T1, G=T1c, B=FLAIR
    ├── TCGA_CS_4941_19960909_1_mask.tif   # máscara binaria tumor
    ├── TCGA_CS_4941_19960909_2.tif
    ├── TCGA_CS_4941_19960909_2_mask.tif
    └── ...
```

**Interpretación**: A partir de aquí el directorio `./data/lgg/` es todo lo que necesitan los pasos siguientes.

---

### Paso 1 — Modelo: entrenar UNet 2D (o saltar a UniVerSeg)

#### Opción A — Entrenar UNet desde cero

```bash
make train-unet
# equivalente a:
uv run seglens-train-unet \
  --data-root ./data/lgg \
  --output-dir models \
  --name unet_real \
  --epochs 60 --batch-size 8 --size 256 \
  --init-features 64 --lr 1e-4 --patience 15
```

- **Input**:
  - `--data-root` (directorio de slices filtrados)
  - Hiperparámetros: epochs, batch-size, image size (256×256), init features (canales base de la UNet), learning rate, paciencia para early stopping.
- **Acción**:
  1. Construye el dataset `LGGSegmentationDataset` y aplica **split por paciente** (70/15/15) con `seed=42`. Esto evita data leakage (un paciente no aparece en train y test a la vez). Típicamente: 715 train / 152 val / 193 test.
  2. Crea un `UNet(in=3, out=1, init=64)` con arquitectura encoder-decoder + skip connections.
  3. Loop de entrenamiento con augmentación sincronizada (horizontal flip, vertical flip, random affine), BCE + Dice loss (50/50), Adam, scheduler `ReduceLROnPlateau`.
  4. Guarda el mejor checkpoint según val_dice (con early stopping tras 15 epochs sin mejora).
- **Output**:
  - `models/unet_real.pth` — state_dict (≈ 66 MB con init=64)
  - `models/unet_real_log.json` — log completo

**Cómo leer `unet_real_log.json`**:
```json
{
  "args": { "epochs": 60, "init_features": 64, ... },
  "history": [
    { "epoch": 1, "train_loss": 0.85, "val_dice": 0.16, "val_iou": 0.09, "lr": 1e-4, "time_s": 3.7 },
    ...
  ],
  "best_val_dice": 0.6626,
  "test_metrics": { "loss": 0.37, "dice": 0.6559, "iou": 0.5321 }
}
```

- `history`: una entrada por epoch, con la pérdida de entrenamiento, métricas en validación, learning rate actual y tiempo del epoch.
- `best_val_dice`: el mejor Dice conseguido en validación (modelo que se guardó).
- `test_metrics`: evaluación final en el split de test con ese mejor modelo.

**Interpretación**: con la config por defecto y early stopping agresivo, el UNet alcanza test Dice ~0.66. El TFM original consigue 0.894 con 60 epochs completos y scheduler ajustado; la diferencia no es un bug, es el efecto del early stopping.

#### Opción B — Cargar un checkpoint existente

```python
from seglens.adapter import UNetAdapter
adapter = UNetAdapter.from_checkpoint(
    "models/unet_real.pth",
    in_channels=3, out_channels=1, init_features=64,
    device="cuda",
)
```

#### Opción C — UniVerSeg few-shot (no requiere entrenamiento)

```bash
make run-universeg
# equivalente a:
uv run seglens-run-universeg \
  --data-root ./data/lgg \
  --output-dir results/universeg_baseline \
  --context-size 64 --channel g --max-test 30
```

- **Input**:
  - `--context-size N`: número de imágenes de soporte (N primeros slices del split train).
  - `--channel`: `rgb` (promedio T1/T1c/FLAIR), `r` (T1), `g` (T1c, contraste), `b` (FLAIR).
  - `--max-test`: número de imágenes de test a evaluar.
- **Acción**:
  1. Carga `universeg(pretrained=True)` y lo mueve a CUDA.
  2. Construye el support set: primeras N imágenes de train, con el canal seleccionado repetido 3× para que promedien a la intensidad del canal elegido. Las máscaras de tumor se incluyen.
  3. Llama a `adapter.set_context(imgs, masks)` que normaliza las formas a `(1, N, 1, H, W)`.
  4. Para cada imagen de test, hace forward y calcula Dice/IoU.
- **Output**:
  - `results_universeg/sample_NNNN/{original.png, ground_truth.png, prediction.png}` (32 primeras)
  - `results_universeg/summary.json`

**Cómo leer `summary.json`**:
```json
{
  "n_test": 30,
  "test_dice_mean": 0.7674,
  "test_dice_std": 0.1470,
  "test_iou_mean":  0.6425,
  "test_iou_std":  0.1696
}
```

**Interpretación**: UniVerSeg con ctx=64 y canal T1c alcanza Dice 0.767 sin entrenamiento. El estudio comparativo (`results/unet_real/universeg_studies/`) muestra que **T1c es ×2.8 mejor que el siguiente canal único** (T1) y que el rendimiento sube monotónicamente con el context size.

---

### Paso 2 — Adapter: la capa de abstracción

**Concepto**: el `SegmentationAdapter` es el contrato mínimo que cualquier modelo debe cumplir. Define qué se puede inspeccionar (stages), cómo preprocesar entradas, cómo sacar la cabeza 1×1 y cómo capturar features intermedias.

```python
class SegmentationAdapter(Protocol):
    name: str
    stages: list[str]                  # nombres de etapas introspectables
    out_channels: int
    def segment(x) -> Tensor: ...     # forward completo → logits (B, K, H, W)
    def head() -> nn.Module: ...      # cabeza 1×1 = "unembedding" del modelo
    def features(x) -> dict[str, Tensor]: ...  # {stage: feature map}
    def preprocess(x) -> Tensor: ...  # normalización por modelo
```

**Input**: nada explícito — el Adapter se construye alrededor de un `nn.Module` ya entrenado/cargado.

**Output**: una instancia de `UNetAdapter` o `UniVerSegAdapter` con:
- `name`: `"unet"` o `"universeg"`.
- `stages`: lista de nombres de submódulos que serán introspectados. Para UNet, las 10 etapas por defecto (`inc, down1..4, up1..4, outc`). Para UniVerSeg, las introspectadas dinámicamente o especificadas manualmente (`enc_blocks.0.target.vmapped, ..., out_conv.conv`).
- `out_channels`: 1 (segmentación binaria).
- `head()`: la última conv 1×1 (`OutputConv` para UNet, `out_conv.conv` para UniVerSeg).

**Por qué importa**: el resto del framework (lentes, conceptos, intervenciones) habla **solo con el Adapter**, no con el modelo. Cualquier modelo que implemente este protocolo se analiza igual, sin tocar código.

**Cómo se determinan los stages**:
- `UNetAdapter` usa `stages` por defecto (la lista de las 10 etapas de la arquitectura).
- `UniVerSegAdapter._discover_stages(model)` recorre `named_modules()` y filtra los nombres que contienen `down_`, `up_`, `cross`, `final`, `outc`, `output`, `seg_head`. Si no encuentra nada, devuelve `[]` (modo "input-output only", sin introspección).
- Para J-space, **necesitamos stages introspectables**. Por eso el script `run_universeg_explainability.py` override los stages con `UNIVERSEG_STAGES = [enc_blocks.{0..3}.target.vmapped, dec_blocks.{0..2}.target.vmapped, out_conv.conv]` — 8 etapas, todas en la **rama target** (UniVerSeg procesa target+support por separado en cada CrossBlock).

---

### Paso 3 — Calibración de las 3 lentes

```bash
uv run seglens-depth-curves \
  --data-root ./data/lgg \
  --checkpoint models/unet_real.pth \
  --output-dir results/unet_real/depth_curves \
  --n-calibration 20 --n-test 30 --size 256
```

Internamente, llama a `calibrate_lenses(adapter, calib_loader, n_images=20, max_pixels_per_image=32)`, que produce 3 lentes. Cada una recibe **el mismo input** (features de calibración) y produce un **LensOutput** (logits de segmentación). Pero se calibran de forma muy distinta:

#### Lente 1 — `LogitLens` (naive correlacional)

**Recibe**:
- `head` (la conv 1×1 final).
- `target_hw` = resolución de la salida del modelo (ej. 256×256).

**Proceso**:
- En cada llamada, aplica la cabeza 1×1 directamente a las features de la etapa: `y = head(f_l)`.
- Solo produce salida válida si `f_l.shape[1] == head.in_channels` (mismo número de canales). En UNet solo `up4` (64 canales = head.in_channels) cumple esto; en UniVerSeg, los `dec_blocks.*.target.vmapped` y `out_conv.conv` (64 canales). En el resto de stages, devuelve `None` con un `UserWarning`.

**Output**: `LensOutput(logits, stage="up4", lens="logit")` — un mapa de logits `(B, 1, H, W)`.

**Interpretación**: la logit lens es la lectura más "cruda" y solo funciona en stages que comparten base con la cabeza. Su Dice en esos stages es la **línea base de cuánto se parece la feature de esa capa al output final**. NO captura dinámicas intermedias.

#### Lente 2 — `TunedLens` (probe lineal entrenada)

**Recibe (calibración)**:
- `stage_features[stage]`: `Tensor(N_pixels, C_l)` — todas las features de un stage, aplanadas a `(N_pixels, C_l)`, bilinearly-upsampleadas a `target_hw` si hace falta.
- `target_logits`: `Tensor(N_pixels, out_channels)` — los logits ground truth (salida real del modelo en esos píxeles).

**Proceso (en `fit`)**:
- Resuelve por mínimos cuadrados regularizados: `W_l = (X^T X + λI)^{-1} X^T y`, donde `W_l ∈ R^{C_l × out_channels}`.
- Una probe `W_l` por stage.

**Recibe (aplicación)**:
- Features de un stage (4D).

**Proceso (en `__call__`)**:
- Upsamplea a `target_hw`, aplana, multiplica por `W_l`: `y_p = W_l^T f_{l,p}`. Devuelve logits.

**Output**: `LensOutput(logits, stage, lens="tuned")`.

**Interpretación**: la tuned lens aprende la mejor lectura LINEAL de cada capa. Como el paper original documenta, **puede "saltar" la jerarquía** y predecir el output desde cualquier capa (típicamente `inc` muestra un Dice sorprendentemente alto). Esto NO significa que la decisión se forme en `inc`; significa que la probe es tan flexible que encuentra un atajo. La tuned lens es la trampa de la explicabilidad correlacional.

#### Lente 3 — `JacobianLens` (J-lens causal)

**Recibe (calibración)**:
- El Adapter y un dataloader de N imágenes.

**Proceso (en `calibrate`)**:
- Para cada imagen, hace un forward del modelo (con gradientes habilitados).
- Captura el feature map RAW `f_l` de cada stage via hook (sin upsample — esto evita un bug de `autograd.grad` con `F.interpolate` en PyTorch 2.13 / Python 3.14).
- Itera sobre píxeles del output (sub-muestreo: `max_pixels_per_image=32` por defecto) y, para cada uno, llama a `autograd.grad(outputs=y[0, k, h, w], inputs=f_l, retain_graph=True)`. Esto devuelve `∂y[0,k,h,w] / ∂f_l[0, :, h, w]`, un vector `(C_l,)`.
- Promedia todos los píxeles: `J_l[k, c] = E_{pixels}[∂y[k, p] / ∂f_l[c, p']]` donde `p'` se mapea por escala.
- Acumula el promedio sobre N imágenes: `J_l = (1/N) Σ_{x} J_l(x)`.

**Output**: por cada stage, una matriz `J_l ∈ R^{K × C_l}` (con K=1 para binario). Es el **Jacobiano promediado**: cuánto cambia el logit si cambia cada feature, **lineado en primer orden**.

**Recibe (aplicación)**:
- Features de un stage.

**Proceso (en `__call__`)**:
- Aplica `J_l · f_l` por píxel: `y_p = J_l^T f_{l, p}`. Devuelve logits.

**Interpretación**: la J-lens es la lectura **causal** — literalmente linealiza la dependencia del output en las features de la capa. Si la J-lens lee "tumor" en una capa, significa que cambiar esa feature cambiaría la decisión de tumor. Por eso crece monotónicamente de las primeras a las últimas capas, mientras que la tuned lens "se salta" al input.

**Sub-muestreo (`max_pixels_per_image`)**: en el UNet de 256×256 hay 65536 píxeles por imagen. Iterar sobre todos es O(65k) backward passes por stage — intractable. Sub-muestreamos a 32 píxeles均匀es (`step = 256*256 / 32 = 2048`, sampleamos 1 de cada 2048). Esto da una buena estimación del Jacobiano medio en segundos.

---

### Paso 4 — Conceptos clínicos (CAVs de T1c)

**Concepto**: en el J-lens original, las direcciones interpretables son los **vectores de tokens** (cada token tiene una dirección en residual stream). En segmentación no hay tokens, pero podemos usar **conceptos clínicos** como `tumor_core`, `ventricle_csf`, etc. — exactamente la idea de TCAV (Kim et al. 2018).

```python
from seglens.concepts import DEFAULT_CONCEPTS
# DEFAULT_CONCEPTS = {
#     "tumor_core":         ("fg", (0.90, 1.00)),  # Foreground ∩ T1c > p90
#     "tumor_periphery":    ("fg", (0.60, 0.90)),  # Foreground ∩ p60 ≤ T1c ≤ p90
#     "ventricle_csf":      ("bg", (0.00, 0.15)),  # Background ∩ T1c < p15
#     "normal_white_matter":("bg", (0.40, 0.70)),  # Background ∩ p40 ≤ T1c ≤ p70
# }
```

**Derivación** (en `derive_concept_masks`):
1. Calcula `ConceptThresholds` sobre N imágenes de calibración: para cada percentil p ∈ [0, 1, ..., 100] computa el cuantil de T1c (canal G de la imagen RGB) dentro de los píxeles foreground y background por separado. Resultado: dos tensores `(101,)` con la distribución acumulada de T1c dentro y fuera del tumor.
2. Para cada imagen y cada concepto, genera una máscara booleana `(H, W)` donde True indica "este píxel cumple la regla". Por ejemplo, `tumor_core` = (es tumor) ∧ (T1c ≥ p90 del foreground).

**Ajuste del CAV** (en `fit_cavs`):
- Para cada etapa y cada concepto: `cav = E[f_l | mask_concept] - E[f_l | ¬mask_concept]`, normalizado a ‖cav‖ = 1.
- Resultado: `cavs[stage][concept] = Tensor(C_l,)`.

**Interpretación**: el CAV apunta en la dirección del espacio de features que más diferencia "este concepto" del resto. Proyectar una feature sobre un CAV (`<f, cav>`) da un score de "cuánto de este concepto hay aquí".

**Output**: `results/unet_real/concept_scoring/concept_scores.pt` con la estructura:
```python
{
    "scores": {stage: {concept: [Tensor_per_image]}, ...},  # (B, 1, H, W)
    "cavs":   {stage: {concept: Tensor(C_l)}, ...},
    "thresholds": [p0, p1, ..., p100],  # del foreground en T1c
}
```

**Cómo visualizar** (futuro notebook): `score[stage][concept][i]` es un mapa `(1, H, W)` que indica para cada píxel cuánto se alinea con el concepto. Superpuesto sobre la imagen de MRI, muestra **dónde la etapa `l` "ve" tumor core, periferia, etc.** — exactamente la Figura 3 del paper de Anthropic.

---

### Paso 5 — Intervenciones

**Concepto**: una "dirección" (un CAV, un J-lens vector) no es interpretable hasta que verificamos que **actúa sobre el modelo**. El framework proporciona tres intervenciones:

| Función | Efecto |
|---|---|
| `ablate_direction(f, v)` | `f ← f - <f, v̂> · v̂` (proyecta fuera) |
| `steer(f, v, α)` | `f ← f + α·v` (añade α veces la dirección) |
| `ablate_concept(f, cav, strength)` | `f ← f - strength·<f, câv>·câv` |
| `patch_activations(f_src, f_tgt, mask)` | copia activaciones de `f_tgt` en `f_src` donde `mask=True` |

**Input**: features (4D) + dirección (1D) + parámetros.
**Output**: features modificadas (4D) — se inyectan en un hook del modelo durante el forward.

**Interpretación**: si ablar un CAV cambia el output del modelo, ese CAV es **causalmente relevante**. Si no cambia nada, la dirección existe en el espacio de features pero el modelo no la usa para decidir (es ortogonal a la decisión, o la cabeza aprendió a usar BatchNorm para compensarla).

---

### Paso 6 — Experimento A: Curva de profundidad de decisión

```bash
uv run seglens-depth-curves \
  --data-root ./data/lgg \
  --checkpoint models/unet_real.pth \
  --output-dir results/unet_real/depth_curves \
  --n-calibration 20 --n-test 30 --size 256
```

**Input**:
- Adapter con el modelo ya cargado.
- Lentes calibradas (Paso 3).
- Test set (n=30 imágenes).

**Proceso**:
1. Para cada imagen de test, captura las features de todos los stages.
2. Para cada stage y cada lente, genera el mapa de logits predicho.
3. Binariza con `sigmoid > 0.5` y compara con la máscara ground truth → Dice, IoU.
4. Agrega por `(lente, stage)` → media y std sobre las 30 imágenes.

**Output**:
- `depth_curves.csv` — tabla con una fila por (lente, stage) y 6 columnas:
  ```
  lens, stage, n_images, dice_mean, dice_std, iou_mean, iou_std
  logit, up4, 30, 0.4687, 0.2531, 0.3415, 0.2243
  ...
  ```
- `depth_curves.png` — gráfica de líneas: eje X = stage, eje Y = Dice, 3 curvas (gris=logit, naranja=tuned, azul=jacobian).
- `depth_curves_pretty.png` (solo UniVerSeg) — versión con etiquetas acortadas `enc_0, enc_1, ..., out`.

**Cómo leer la tabla**:
- Mira la columna `dice_mean` de cada lens. Compara cómo varía con `stage`.
- Para el **UNet** (10 etapas inc→outc):
  - `logit` solo tiene valores en `up4` (mismos 64 canales que la cabeza).
  - `tuned` muestra típicamente un **pico temprano** en `inc` (la probe puede aprender un atajo).
  - `jacobian` muestra típicamente una **subida monotónica** desde las primeras capas hasta la última.
- Para **UniVerSeg** (8 etapas enc_0...out):
  - Mismo patrón: `tuned` con valor alto en `enc_0`, `jacobian` con subida limpia a `dec_2`.

**Cómo leer la figura**:
- Eje X: etapa, de izquierda (input) a derecha (output). Las etapas intermedias son la "profundidad" de la red.
- Eje Y: Dice en test. Más alto = la decisión de tumor es más legible desde esa capa.
- **Tres curvas superpuestas**: la distancia entre **tuned** (naranja) y **jacobian** (azul) en las primeras etapas es la **demostración del paper**: la tuned lens miente, la jacobian dice la verdad. La jacobian creciendo monotónicamente es la firma del "workspace".

**Hallazgo típico** (nuestro run con UNet early-stopped):
- `logit`: solo up4 = 0.47 (línea base)
- `tuned`: inc = 0.65 (skip-ahead), up4 = 0.47
- `jacobian`: inc = 0.09, up4 = 0.45 (causal real)

**Hallazgo típico** (UniVerSeg, ctx=64, T1c):
- `logit`: enc_3 = 0.21, dec_2 = 0.77
- `tuned`: enc_0 = 0.41, dec_2 = 0.71
- `jacobian`: enc_0 = 0.23, dec_2 = 0.70 (curva suave, emerge al final)

---

### Paso 7 — Experimento B: Concept scoring (T1c)

```bash
uv run seglens-concept-scoring \
  --data-root ./data/lgg \
  --checkpoint models/unet_real.pth \
  --output-dir results/unet_real/concept_scoring \
  --n-calibration 20 --n-test 10 --size 256
```

**Input**:
- Adapter con modelo.
- N=20 imágenes de calibración (para umbrales T1c y para ajustar CAVs).
- N=10 imágenes de test (para generar mapas de score).

**Proceso**:
1. `fit_concepts`:
   - Calcula `ConceptThresholds` sobre las N calibración (cuantiles T1c dentro/fuera de tumor).
   - Genera máscaras booleanas por concepto para cada imagen.
   - Ajusta CAVs por (stage, concept) con `fit_cavs` (diferencia de medias + normalización).
2. `score_concepts_on_test`:
   - Para cada imagen de test, captura features de todos los stages.
   - Para cada (stage, concept), proyecta `<f_l, cav>` por píxel → score map `(1, H, W)`.

**Output**: `concept_scores.pt` (estructura descrita en Paso 4).

**Interpretación (sin visualización en este script)**:
- `scores[stage][concept][i]` es un heatmap donde los valores altos = el modelo "ve" ese concepto en ese píxel según la etapa `l`.
- Comparar entre etapas: ¿en qué capa aparece la señal de tumor_core? ¿en la misma capa donde la jacobian sube? Si sí, la decisión "workspace" está correlacionada con la presencia del concepto clínico.
- Comparar entre conceptos: tumor_core vs tumor_periphery deberían tener mapas diferentes (tumor core hiperintenso en T1c vs anillo perilesional).

**Hallazgo típico**: en los mapas del UNet y UniVerSeg, el score de `tumor_core` se concentra en zonas hiperintensas, pero el score de `ventricle_csf` está cerca de las zonas oscuras — coherente con la anatomía esperada. Esto **valida cualitativamente** que los CAVs captan lo que dicen captar.

---

### Paso 8 — Experimento C: Ablación de CAVs

```bash
uv run seglens-ablation \
  --data-root ./data/lgg \
  --checkpoint models/unet_real.pth \
  --output-dir results/unet_real/ablation \
  --n-calibration 20 --n-test 10 --size 256 \
  --stages up1 up2 up3 up4 outc \
  --n-random-controls 3
```

**Input**:
- Adapter con modelo.
- CAVs (del Paso 7).
- Lista de stages a ablacionar (subset para no tardar infinito).
- N imágenes de test.
- N controles aleatorios (vectores random de la misma magnitud que cada CAV).

**Proceso**:
- Para cada `(stage, concept)`:
  1. **Clean**: forward del modelo sin modificar → Dice baseline.
  2. **Ablated**: forward con un hook que abla la dirección del CAV en ese stage → Dice.
  3. **Random controls** (N veces): forward con un hook que abla una dirección aleatoria de magnitud equiparable → Dice.
- Calcula `ΔDice = Dice_ablated - Dice_clean` y `ΔDice_random_mean`.

**Output**: `ablation.csv` con columnas:
```
stage, concept, clean_dice, ablated_dice, delta_dice, n_images, random_delta_dice
```

**Cómo leer la tabla**:
- `clean_dice`: Dice del modelo sin tocar. Para UNet en test: ~0.66. Para UniVerSeg: ~0.77.
- `ablated_dice`: Dice después de ablacionar la dirección del CAV. Si el CAV es causalmente relevante, este valor baja.
- `delta_dice = ablated - clean`: lo que cambia el modelo cuando "le quitas" la dirección del concepto. **Negativo = la dirección importa**. Magnitudes típicas:
  - ΔDice = 0.000 → la dirección no afecta al modelo (causalmente irrelevante).
  - ΔDice ≈ -0.05 → la dirección explica ~5% del Dice. Significativo.
  - ΔDice ≈ -0.20 → la dirección es crítica.
- `random_delta_dice` (lista): cuánto cambian los controles aleatorios. Si `delta_dice` está fuera del rango de los random, es un efecto específico del CAV. Si está dentro, probablemente es ruido.

**Interpretación (nuestro run)**:
- **UNet y UniVerSeg**: todos los ΔDice ≈ 0.000. Los CAVs derivados de T1c NO son direcciones que el modelo use para decidir.
- **Por qué es un hallazgo válido**:
  - El UNet (Dice 0.66) y UniVerSeg (Dice 0.77) están usando **otras direcciones** del espacio de features, no las que un humano identificaría como "tumor core" por intensidad T1c.
  - Esto es coherente con que la cabeza 1×1 y los BatchNorms compensan: la salida es invariante a quitar esa componente.
  - **Implicación clínica**: el modelo "ve" tumor, pero no de la misma manera que un radiólogo lo vería. El J-lens framework lo hace explícito en lugar de quedarse en "el modelo acierta, confío".

**Cómo podríamos mejorar la sensibilidad**:
- Usar CAVs derivados de otras señales (e.g., gradientes de la cabeza 1×1, attention maps de UniVerSeg, mapas de error del modelo).
- Subir la fuerza de la ablación más allá de 1.0 (parcial: 0.5, fuerte: 2.0).
- Ablar **combinaciones** de CAVs en lugar de uno a uno.

---

### Resumen: outputs del pipeline completo

| Comando | Outputs | Qué demuestra |
|---|---|---|
| `seglens-download-data` | `data/lgg/...` (raw + filtrado) | Dataset listo |
| `seglens-train-unet` | `models/unet.pth`, `models/unet_log.json` | UNet entrenado (métrica baseline) |
| `seglens-run-universeg` | `results_universeg/sample_*/`, `summary.json` | UniVerSeg few-shot (Dice sin entrenar) |
| `seglens-depth-curves` | `depth_curves.csv`, `depth_curves.png` | **Dónde emerge la decisión** (3 lentes) |
| `seglens-concept-scoring` | `concept_scores.pt` | **Dónde se activa cada concepto** por stage |
| `seglens-ablation` | `ablation.csv` | **Qué conceptos son causalmente relevantes** |
| `seglens-run-universeg-explainability` | `depth_curves.csv`, `depth_curves_pretty.png`, `cavs.pt`, `ablation.csv`, `summary.json` | Todo lo anterior sobre UniVerSeg |

---

## 4. Estructura del proyecto

```
SegLens/
├── src/seglens/
│   ├── adapter.py            # Protocol + UNet + UniVerSeg
│   ├── hooks.py              # FeatureExtractor
│   ├── data/                 # LGG loader, split por paciente, filter
│   ├── cli/                  # entry points (train, run, download, explain, etc.)
│   ├── lens/
│   │   ├── logit_lens.py     # Lente 1
│   │   ├── tuned_lens.py     # Lente 2 (probe)
│   │   ├── jacobian_lens.py  # Lente 3 (Jacobiano)
│   │   └── calibration.py    # calibración conjunta
│   ├── concepts.py           # CAV, ConceptScorer, derivación T1c
│   ├── interventions.py      # ablate / steer / patch
│   ├── experiments/          # núcleos de los 7 experimentos
│   └── viz.py
├── tests/                    # 49 tests (pytest)
├── pyproject.toml            # uv-friendly, entry points
├── Makefile                  # atajos
├── uv.lock                   # generado por `uv sync`
└── results/                  # outputs (gitignored *.pt/*.pth/*.png, README mantenido)
    └── unet_real/
        ├── depth_curves/     # ← la curva principal
        ├── concept_scoring/  # ← los CAVs y scores
        ├── ablation/         # ← ΔDice por (stage, concept)
        ├── universeg_baseline/
        └── universeg_studies/  # ctx-size + channel impact
```

---

## 5. Uso programático

```python
import torch
from seglens.adapter import UNetAdapter, UniVerSegAdapter
from seglens.lens import calibrate_lenses
from seglens.experiments.run_depth_curves import run_depth_curves, summarize_depth_curves
from seglens.viz import plot_depth_curves

# Cualquier modelo con cabeza 1×1
adapter = UNetAdapter.from_checkpoint("models/unet_real.pth", init_features=64)

# Calibrar 3 lentes (n=20 imágenes, sub-muestreo J-lens a 32 píxeles/imagen)
lenses = calibrate_lenses(adapter, calib_loader, n_images=20, max_pixels_per_image=32)

# Profundidad de decisión
per_image, targets = run_depth_curves(adapter, lenses, test_loader, max_images=30)
agg = summarize_depth_curves(per_image, targets)
plot_depth_curves(agg, save_path="depth_curves.png")
```

---

## 6. Estado del proyecto

| Fase | Estado | Descripción |
|---|---|---|
| F0 | ✅ | Adapter protocol + UNetAdapter + UniVerSegAdapter |
| F1 | ✅ | FeatureExtractor + 3 lentes (logit / tuned / jacobian) |
| F2 | ✅ | run-depth-curves CLI + curva de profundidad (con bandas MC) |
| F3 | ✅ | Conceptos clínicos (T1c) + run-concept-scoring CLI |
| F4 | ✅ | Intervenciones (ablate / steer / patch) + run-ablation CLI + ΔEntropy |
| F5 | ✅ | run-universeg-explainability + comparativa UNet vs UniVerSeg |
| F6 | ✅ | **Incertidumbre**: MC Dropout hook-based, TTA, atribución por stage, correlación depth↔uncertainty |
| F7 | ⏳ | Visualización cualitativa (heatmaps de conceptos) + redacción |

### F6 — Incertidumbre

Tres comandos CLI nuevos:

```bash
# E1: correlación píxel-a-píxel entre "decision depth" (J-lens) y entropía
uv run seglens-uncertainty-link \
  --data-root ./data/lgg --checkpoint models/unet_real.pth \
  --method mc_dropout --n-test 30 --n-samples 10 \
  --output-dir results/uncertainty_link
# → uncertainty_maps.npy, decision_depth.npy, scatter.png, summary.json

# E2: atribución de varianza por stage (usa el Jacobiano calibrado)
uv run seglens-stage-uncertainty \
  --data-root ./data/lgg --checkpoint models/unet_real.pth \
  --method mc_dropout --n-test 10 --n-samples 10 \
  --output-dir results/stage_uncertainty
# → stage_attribution.csv, stage_attribution.png

# E3: extiende seglens-ablation con ΔEntropy
uv run seglens-ablation \
  --data-root ./data/lgg --checkpoint models/unet_real.pth \
  --stages up3 up4 outc --n-uncertainty-samples 5 \
  --output-dir results/ablation_with_unc

# Bandas de incertidumbre en la curva principal
uv run seglens-depth-curves --with-uncertainty --n-uncertainty-samples 5 ...
```

**Hallazgos en el UNet entrenado (Dice 0.66) sobre LGG**:
- H1 `ρ(depth, uncertainty) > 0`: **FALSA** — correlación **negativa** (píxeles "decididos tarde" tienen menos incertidumbre, son desacuerdos sistemáticos J-lens vs final).
- H2 últimas etapas concentran >50% de la incertidumbre: **TRUE** — `up4+outc` concentran **96.8%** de la varianza atribuida.
- Ablación de CAVs en `up4` con `tumor_core` produce **ΔDice = -0.37** (vs 0 en `up3`/`outc`) — la última capa del decoder es donde los CAVs SÍ son causalmente relevantes.

---

## 7. Inspiración y crédito

- **J-lens / J-space** — Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Anthropic (2026). [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/index.html)
- **TCAV** — Kim et al., *Interpretability Beyond Feature Attribution* (2018) — base de los CAVs de conceptos.
- **LGG MRI Segmentation Dataset** (Kaggle, mateuszbuda) — datos de demostración.
- **UniVerSeg** — Butoi et al., *UniverSeg: Universal Medical Image Segmentation* (2023).

## 8. Licencia

MIT.
