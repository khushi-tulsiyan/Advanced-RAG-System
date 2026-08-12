.PHONY: help install install-ml test lint format index evaluate serve clean

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python  ## Install core deps (no torch) + dev tools
	$(BIN)/pip install -e '.[faiss,dev]'

install-ml: install  ## Additionally install the real embedding/reranking models
	$(BIN)/pip install -e '.[ml]'

test: ## Run the test suite (offline, no model downloads)
	$(BIN)/pytest

lint: ## Check formatting and lint rules
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

format: ## Apply formatting and autofixes
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

index: ## Build the retrieval index from data/raw_docs
	$(BIN)/python scripts/build_index.py

evaluate: ## Measure retrieval quality against data/eval/qrels.jsonl
	$(BIN)/python scripts/evaluate.py --compare

serve: ## Run the API at http://127.0.0.1:8000
	cd src && ../$(BIN)/uvicorn rag.app:app --reload --port 8000

clean: ## Remove generated index artefacts and caches
	rm -rf data/vector_store data/processed_chunks
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
