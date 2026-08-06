# SegLens

> **Lentes causales para explicabilidad agnóstica del modelo en segmentación de imagen médica.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)

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

## ¿Por qué?

El J-lens responde: ¿en qué capa "decide" el modelo qué token producir? En segmentación la pregunta análoga es: ¿en qué capa "decide" el modelo qué píxel es tumor? Y la respuesta tiene impacto clínico:

- **Auditoría del modelo**: ¿la decisión se forma donde debería (cuello de botella semántico) o en capas superficiales (overfitting a artefactos)?
- **Confianza por píxel**: ¿los píxeles "decididos tarde" son los más inciertos? Conexión directa con mapas de incertidumbre (MC Dropout, TTA).
- **Interpretabilidad clínica**: ¿qué conceptos (realce, edema, ventrículo) están activos en cada píxel de cada capa?
- **Comparativa de modelos**: ¿dónde "decide" un foundation model few-shot vs un UNet entrenado desde cero?

## ¿Cómo se ve un experimento?

```python
import torch
from seglens.adapter import UNetAdapter
from seglens.lens import LogitLens, TunedLens, JacobianLens
from seglens.experiments import run_depth_curves

# 1) Enchufar tu modelo (UNet, UniVerSeg, o el tuyo)
adapter = UNetAdapter.from_checkpoint("unet_model.pth", in_channels=3, out_channels=1)

# 2) Calibrar las lentes con un split de validación
lenses = run_calibration(adapter, val_loader, n_images=200)

# 3) Medir Dice-por-etapa en test
results = run_depth_curves(adapter, lenses, test_loader)
results.plot()  # curva de profundidad de decisión
```

## Estado

**Fase actual**: 0–1 (núcleo del framework). Ver [ROADMAP.md](ROADMAP.md) para el plan completo.

| Fase | Estado | Descripción |
|---|---|---|
| F0 | ✅ | Adapter protocol + UNetAdapter + UniVerSegAdapter |
| F1 | ✅ | FeatureExtractor + 3 lentes (logit / tuned / jacobian) |
| F2 | ⏳ | Curva de profundidad + comparativa de lentes |
| F3 | ⏳ | Conceptos clínicos (T1c) + TCAV scoring |
| F4 | ⏳ | Intervenciones (ablate / steer / patch) |
| F5 | ⏳ | Comparativa cruzada UNet vs UniVerSeg |
| F6 | ⏳ | Integración con incertidumbre |
| F7 | ⏳ | Visualización + redacción |

## Instalación (en desarrollo)

```bash
git clone <repo> && cd SegLens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Estructura

```
SegLens/
├── src/seglens/
│   ├── adapter.py            # Protocol + UNet + UniVerSeg
│   ├── hooks.py              # FeatureExtractor
│   ├── lens/
│   │   ├── logit_lens.py     # Lente 1
│   │   ├── tuned_lens.py     # Lente 2 (probe)
│   │   └── jacobian_lens.py  # Lente 3 (Jacobiano)
│   ├── concepts.py           # CAV, ConceptScorer, derivación T1c
│   ├── interventions.py      # ablate / steer / patch
│   ├── experiments/          # scripts CLI de E1–E7
│   └── viz.py
├── tests/                    # pytest
├── docs/                     # notas, figuras, ROADMAP
└── examples/                 # notebooks
```

## Inspiración y crédito

- **J-lens / J-space** — Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Anthropic (2026). [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/index.html)
- **TCAV** — Kim et al., *Interpretability Beyond Feature Attribution* (2018) — base de los CAVs de conceptos.
- **LGG MRI Segmentation Dataset** (Kaggle) — datos de demostración.

## Licencia

MIT.
