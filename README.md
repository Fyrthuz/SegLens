# SegLens

> **Lentes causales para explicabilidad agnóstica del modelo en segmentación de imagen médica.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet)](https://docs.astral.sh/uv/)

---

## ¿Qué es?

SegLens es un framework de **explicabilidad por etapas** para modelos de segmentación de imagen. Está inspirado en el *Jacobian Lens* (J-lens) de Anthropic —la técnica que identifica el "workspace verbalizable" en LLMs— pero adaptado al dominio de segmentación, donde no hay vocabulario ni unembedding textual.

**Idea nuclear**: para cada etapa intermedia de un modelo de segmentación (UNet, UniVerSeg, o cualquier encoder-decoder), calculamos un *readout lineal* de la decisión de segmentación desde sus features. Comparamos tres lentes:

| Lente | Mapeo | Naturaleza |
|---|---|---|
| **Logit lens** | Aplicar la cabeza 1×1 directamente | Correlacional, naive |
| **Tuned lens** | Probe lineal entrenada por etapa | Correlacional, aprende corrección de base |
| **J-lens** | Jacobiano promediado `E[∂y/∂f_l]` por etapa | **Causal** (lineal de primer orden) |

Más una capa de **conceptos clínicos** (estilo TCAV: tumor core, periferia, ventrículos, tejido normal — derivados de T1c) que actúa como "vocabulario" análogo a los tokens del paper original, y un módulo de **intervenciones** (ablación, steering, patching) para verificar causalmente qué direcciones de features importan.

Todo detrás de un `SegmentationAdapter` minimal, así que **cualquier modelo con cabeza 1×1 puede enchufarse** sin tocar el resto del framework.

## Quick start (con `uv`)

```bash
# clonar e instalar (deps mínimas, sin universeg)
git clone https://github.com/Fyrthuz/SegLens && cd SegLens
make install

# o incluyendo UniVerSeg + dev tools
make dev

# smoke test end-to-end (UNet sintético, las 3 lentes, ~3s)
make run-smoke

# tests
make test
```

## Comandos principales

```bash
# 1) Descargar LGG MRI desde Kaggle (necesita KAGGLE_USERNAME y KAGGLE_KEY)
make download-data

# 2) Entrenar UNet (60 epochs, 1060 slices, 108 pacientes, ~5 min en RTX 30/40)
make train-unet      # → ./models/unet.pth + ./models/unet_log.json

# 3) Inferencia few-shot con UniVerSeg (context-size 64, canal T1c)
make run-universeg   # → ./results_universeg/{sample_*, summary.json}

# 4) Curva de profundidad de decisión (las 3 lentes sobre UNet)
uv run seglens-depth-curves --data-root ./data/lgg --checkpoint ./models/unet.pth

# 5) Ablación de CAVs de conceptos
uv run seglens-ablation --data-root ./data/lgg --checkpoint ./models/unet.pth

# 6) Concept scoring
uv run seglens-concept-scoring --data-root ./data/lgg --checkpoint ./models/unet.pth
```

## Uso programático

```python
import torch
from seglens.adapter import UNetAdapter, UniVerSegAdapter
from seglens.lens import calibrate_lenses
from seglens.experiments.run_depth_curves import run_depth_curves, summarize_depth_curves

# Cualquier modelo con cabeza 1×1
adapter = UNetAdapter.from_checkpoint("models/unet.pth")
lenses = calibrate_lenses(adapter, val_loader, n_images=200)
per_image, targets = run_depth_curves(adapter, lenses, test_loader)
agg = summarize_depth_curves(per_image, targets)
```

## Estado del proyecto

| Fase | Estado | Descripción |
|---|---|---|
| F0 | ✅ | Adapter protocol + UNetAdapter + UniVerSegAdapter |
| F1 | ✅ | FeatureExtractor + 3 lentes (logit / tuned / jacobian) |
| F2 | ✅ | run-depth-curves CLI + curva de profundidad |
| F3 | ✅ | Conceptos clínicos (T1c) + run-concept-scoring CLI |
| F4 | ✅ | Intervenciones (ablate / steer / patch) + run-ablation CLI |
| F5 | ⏳ | Comparativa cruzada UNet vs UniVerSeg (validada manualmente, falta script) |
| F6 | ⏳ | Integración con incertidumbre (MC Dropout / TTA) |
| F7 | ⏳ | Visualización avanzada + redacción |

## Estructura

```
SegLens/
├── src/seglens/
│   ├── adapter.py            # Protocol + UNet + UniVerSeg
│   ├── hooks.py              # FeatureExtractor
│   ├── data/                 # LGG loader, split por paciente, filter
│   ├── cli/                  # entry points (train, run, download, etc.)
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
└── uv.lock                   # generado por `uv sync`
```

## Inspiración y crédito

- **J-lens / J-space** — Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Anthropic (2026). [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/index.html)
- **TCAV** — Kim et al., *Interpretability Beyond Feature Attribution* (2018) — base de los CAVs de conceptos.
- **LGG MRI Segmentation Dataset** (Kaggle, mateuszbuda) — datos de demostración.

## Licencia

MIT.

