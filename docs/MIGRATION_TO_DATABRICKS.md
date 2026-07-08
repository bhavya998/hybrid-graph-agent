# Migration to Databricks / Mosaic AI

This document maps every local-stack component to its Databricks equivalent.
The codebase is structured behind provider interfaces (`LLMClient`, `Embedder`,
`VectorStore`, `GraphStore`, `Reranker`), so swapping is config-driven — no agent
or ingestion logic changes.

## Config swap

Set `HGA_PROFILE=databricks` and implement the databricks branches in each
factory (`hga/llm/__init__.py`, `hga/embeddings/__init__.py`,
`hga/retrieval/__init__.py`, `hga/graph/__init__.py`).

## Component map

| Local (now) | Databricks / Mosaic AI | Migration step |
|---|---|---|
| `AnthropicLLMClient` (direct SDK) | **Foundation Model APIs** via **AI Gateway** | Point `AnthropicLLMClient` at the AI Gateway endpoint URL; or implement `DatabricksLLMClient` using `databricks-sdk` `workspace_client.serving_endpoint` calls. AI Gateway adds rate-limit, routing, audit, cost tracking. |
| `VoyageTextEmbedder` (`voyage-3-large`) | **Foundation Model API** `databricks-text-embedding` or **External Model** (Voyage) registered in AI Gateway | Implement `DatabricksEmbedder` calling the serving endpoint. Same `Embedder` interface. |
| `CohereReranker` (`rerank-v3.5`) | **Foundation Model API** rerank / External Model in AI Gateway | Implement `DatabricksReranker`. Same `Reranker` interface. |
| `QdrantStore` (local file mode) | **Vector Search** (Auto or Direct-Access index) | Implement `DatabricksVectorStore` using `databricks.vector_search_client`. Map collection → Vector Search index, dense vector → embedding source, sparse BM25 → hybrid query mode. Delta table backing the index replaces `chunks.parquet`. |
| `FastembedBM25` (local sparse) | Vector Search **hybrid** mode (built-in BM25) | Vector Search handles BM25 server-side when `embedding_type="hybrid"`. Drop fastembed. |
| `KuzuGraphStore` (embedded) | **Delta tables + GraphFrames** (or Neo4j via container service) | GraphRAG indexer outputs (entities, relationships, communities) land in Unity-Catalog Delta tables. Query via Spark GraphFrames motif queries instead of Cypher. Document the tradeoff: Databricks has no native graph DB. |
| Microsoft GraphRAG indexer (local) | GraphRAG indexer running as a **Databricks Job** / **Workflow** on a serverless cluster | Wrap `GraphragIndexer.run()` in a Job task. Read input from Delta, write output to Delta in Unity Catalog. Use Foundation Model API for LLM calls. |
| `chunks.parquet` (local file) | **Delta table** in Unity Catalog (`hga.chunks`) | `chunks.parquet` → `spark.write.format("delta")`. Unity Catalog governs with row/column ACLs. |
| Langfuse Cloud (tracing) | **MLflow Tracing** + **Lakehouse Monitoring** | Langfuse spans → MLflow traces (auto-logged by Databricks SDK). Lakehouse Monitoring handles drift/alerting. Keep `Tracer` interface, implement `MLflowTracer`. |
| Custom eval harness (RAGAS-style) | **Agent Evaluation** (`databricks.agents.eval`) + **Review App** | Golden set → MLflow evaluation dataset. `hga eval` → `mlflow.evaluate()` with agent-eval metrics. Review App for human judges. |
| `FastAPI` + `Chainlit` | **Model Serving** (Agent endpoint) + **Review App** / **Genie** | Deploy compiled LangGraph via `databricks.agents.deploy()`. The agent becomes a REST endpoint on Databricks. Chainlit UI → Databricks Review App or Genie. |
| MCP tool servers (stdio/local) | **Unity Catalog Functions** + **Jobs** | Each MCP tool → a `CREATE FUNCTION` in Unity Catalog. Agent calls UC functions via function-calling. External connectors → Databricks Jobs/Workflows. |
| `SqliteSaver` (checkpoints) | **Databricks SQL** (managed Postgres-compatible) | LangGraph checkpoint table in Databricks SQL warehouse. |

## Governance

| Concern | Local | Databricks |
|---|---|---|
| Secrets | `.env` / pydantic-settings | Databricks **Secrets** + Unity Catalog |
| Data lineage | none | Unity Catalog automatic lineage |
| Access control | filesystem | UC table/column/row-level ACLs |
| Audit | none | UC audit logs + AI Gateway request logs |
| PII redaction | `hga.observability.redact()` | same + UC dynamic views for column masking |

## Migration order (recommended)

1. **LLM + Embeddings + Reranker** → Foundation Model APIs via AI Gateway (lowest risk, biggest governance win)
2. **Vector store** → Vector Search (chunks table → Delta → VS index)
3. **Eval** → `databricks.agents.eval` + MLflow datasets
4. **Graph** → Delta + GraphFrames (entities/relationships as UC tables)
5. **Serving** → `databricks.agents.deploy()` Model Serving endpoint
6. **Tracing** → MLflow Tracing (replace Langfuse spans)
7. **MCP tools** → Unity Catalog Functions

Each step is independently shippable because of the interface-first architecture.
