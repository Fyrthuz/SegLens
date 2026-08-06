.PHONY: help install sync dev test lint run-smoke train-unet run-universeg download-data clean uncertainty-link stage-uncertainty

PYTHON ?= uv run python
SEGLENS ?= uv run

help:
	@echo "SegLens — make targets:"
	@echo "  make install     uv sync (deps mínimas)"
	@echo "  make dev         uv sync --extra universeg (incluye universeg)"
	@echo "  make test        pytest"
	@echo "  make lint        ruff"
	@echo "  make run-smoke   ejecuta smoke_unet end-to-end"
	@echo "  make download-data   descarga + filtra LGG (~1GB)"
	@echo "  make train-unet     entrena UNet sobre data/lgg"
	@echo "  make run-universeg  ejecuta UniVerSeg few-shot"

install:
	uv sync

dev:
	uv sync --extra universeg --extra dev

test:
	uv run --extra dev pytest tests/

lint:
	uv run --extra dev ruff check src tests

run-smoke:
	$(SEGLENS) seglens-smoke

train-unet:
	$(SEGLENS) seglens-train-unet --data-root ./data/lgg --epochs 60 --output-dir ./models

run-universeg:
	$(SEGLENS) seglens-run-universeg --data-root ./data/lgg --context-size 64 --channel g

download-data:
	$(SEGLENS) seglens-download-data --raw ./data/lgg_raw --filtered ./data/lgg

uncertainty-link:
	$(SEGLENS) seglens-uncertainty-link --data-root ./data/lgg --checkpoint ./models/unet_real.pth --output-dir ./results/uncertainty_link

stage-uncertainty:
	$(SEGLENS) seglens-stage-uncertainty --data-root ./data/lgg --checkpoint ./models/unet_real.pth --output-dir ./results/stage_uncertainty

clean:
	rm -rf .venv .pytest_cache dist build src/*.egg-info src/seglens.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
