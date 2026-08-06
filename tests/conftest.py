"""Pytest config."""

import sys
from pathlib import Path

# Asegurar que ``src/`` está en el path cuando se ejecuta ``pytest`` sin instalar.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
