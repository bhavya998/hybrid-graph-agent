# AGENTS.md — contract for any AI coding agent working on this repo

Non-negotiable rules. Follow exactly.

## Environment
- Python 3.12 only (pinned via `.python-version`). Use `uv` for all package ops.
- Run anything via `uv run <cmd>` so the venv is used. Never `pip install`.
- Default stack (`local_gpu`): self-hosted bge-small, bge-reranker and Qwen 1.5B 4-bit. The opt-in `local` profile is all-API: Anthropic (LLM), Voyage (embeddings), Cohere (rerank), Langfuse Cloud (tracing). No Docker.

## Architecture contracts
1. **Interfaces first.** Every external capability is behind a Protocol/ABC in `hga/llm/`, `hga/embeddings/`, `hga/retrieval/`, `hga/graph/`, `hga/tools/`. Agent/graph code imports interfaces, never provider SDKs directly.
2. **Provider profiles.** `hga.config.Settings` has a `profile: Literal["local_gpu","local","databricks"]`. Concrete implementations are selected by profile. `local_gpu` (default) and `local` are implemented; the databricks profile is a stub documented in `docs/MIGRATION_TO_DATABRICKS.md`.
3. **Pydantic v2 everywhere.** `Chunk`, `AgentState`, `Citation`, `Settings`, all LLM I/O schemas. Mypy strict passes.
4. **Type hints mandatory** on every function (Python type hints, per global rule).
5. **Docstrings mandatory** on every public function/class (per global rule).

## Observability contract
6. **Every LangGraph node logs a Langfuse span** carrying `trace_id` from `AgentState`. Use the Langfuse decorator/context manager, not bare prints.
7. **No PII or secret logging.** Secrets live only in `.env` (loaded via `pydantic-settings`). A redaction filter runs before any payload hits Langfuse.

## Data contract
8. **`Chunk` is the single schema** every ingestion handler emits (see `hga/chunking/schema.py`). Nothing else flows into the vector store or graph.
9. **Idempotent ingestion.** `content_hash` dedupes; re-running `hga ingest` updates, doesn't duplicate.

## Retrieval contract
10. **Hybrid = dense + sparse fused via RRF (k=60), then cross-encoder rerank.** Never return raw fused results to the agent; always rerank.
11. **Every retrieved chunk carries provenance** (`doc_id`, `page`, `chunk_id`) so citations are reconstructable.

## Agent contract
12. **Adaptive routing.** A cheap classifier LLM picks one of: `vector | hybrid | graph_local | graph_global | tool | agentic`. The Critic can escalate to graph on low faithfulness (CRAG pattern), max 2 retries.
13. **State is `AgentState`** (TypedDict in `hga/agents/state.py`). No node mutates global state; nodes return partial dicts merged by reducers.

## Eval contract
14. **`eval/golden/golden.jsonl`** is the CI gate. New code must not regress faithfulness/context-recall vs main.
15. **One command runs the eval:** `hga eval --report eval/reports/latest.json`.

## Code style
- No comments unless asked. Let names + docstrings carry intent.
- Ruff + mypy clean before commit. Run via `make lint`.
- One module = one responsibility. If a file exceeds ~400 lines, split.

## Running
- `uv sync` — install
- `hga ingest` — fetch corpus, parse, embed, index (vector + graph)
- `hga serve` — FastAPI + Chainlit
- `hga eval` — run eval harness
- `make lint` — ruff + mypy
