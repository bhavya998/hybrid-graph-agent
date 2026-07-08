.PHONY: help install sync lint format typecheck test serve ingest eval graph clean ui-dev ui-build

help:
	@echo "hga make targets:"
	@echo "  sync        - install Python deps via uv"
	@echo "  ingest      - fetch arxiv+repos, parse, embed, index"
	@echo "  serve       - run FastAPI server"
	@echo "  eval        - run LLM-judge eval harness"
	@echo "  lint        - ruff check + mypy"
	@echo "  format      - ruff format"
	@echo "  test        - pytest"
	@echo "  ui-dev      - start Next.js dev server"
	@echo "  ui-build    - production build of the UI"
	@echo "  clean       - remove data + caches"

sync:
	uv sync --all-extras

ingest:
	uv run hga ingest

serve:
	uv run hga serve

eval:
	uv run hga eval

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest -v

clean:
	rm -rf data/raw data/processed data/qdrant data/graphrag_output data/kuzu data/repos
	find . -type d -name __pycache__ -exec rm -rf {} +

ui-dev:
	cd ui && npm run dev

ui-build:
	cd ui && npm run build
