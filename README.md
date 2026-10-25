<div align="center">

# 🧠 hga — Hybrid RAG + GraphRAG + Multi-Agent

### A fully-local research-to-code assistant. Ask questions across the AI literature and the codebases that implement it — with provenance, a knowledge graph, and an adaptive agent loop.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Working%20prototype-6366f1)]()

**Zero API keys by default — the `local_gpu` profile runs 100% on your GPU.** An optional API profile swaps in Claude, Voyage and Cohere.

</div>

---

## 🎯 What This Is

Most RAG demos are a single vector store + an LLM call. This isn't that.

**hga** is a retrieval system that combines three retrieval paradigms — **hybrid search** (dense + sparse), **GraphRAG** (knowledge graph), and **tool-augmented agents** — into a single adaptive pipeline that routes each query to the best strategy.

You point it at a corpus of arXiv papers + their GitHub repos. You ask questions. It decides whether to use vector search, BM25 keyword search, graph traversal, or MCP tools, retrieves the most relevant context, reranks it with a cross-encoder, and generates a cited answer — all running on local models, no cloud required.

```
"What are the main themes across these papers?"     → GraphRAG (global community search)
"Which authors introduced RAG?"                      → GraphRAG (local entity search)
"How does BM25 compare to dense retrieval?"          → Hybrid (dense + BM25 + rerank)
"Show me the retrieval code in the FLARE repo"       → MCP tool (read_repo_file)
"What is FLARE?"                                      → Vector (dense semantic search)
```

---

## 🎯 Why Not Just Vector Search?

Pure vector RAG fails in specific, predictable ways — and a senior system should know when. Here are concrete examples from our arXiv corpus showing **why each retrieval mode exists**:

### Vector-only fails on exact matches

A query like *"Show me the retrieval code in the FLARE repo"* needs the token `flare` and `retrieval` together in source files. Vector search returns semantically similar chunks ("augmented generation pipeline"…) but **misses the exact file** `src/retriever.py` because the embedding space doesn't preserve precise identifiers. **BM25 sparse retrieval catches it instantly** — exact string match in the chunk content.

### Vector-only fails on relationships

*"Which authors introduced RAG that later papers built on?"* is a multi-hop relational question. No single chunk contains the full author→paper→citation chain. Vector search returns fragments about RAG and fragments about citations, but never connects them. **GraphRAG's local entity search traverses the author→paper→citation edges** in the knowledge graph and reconstructs the answer from the graph topology.

### Vector-only fails on global summarization

*"What are the main research themes across these 40 papers?"* requires aggregating across the entire corpus — no single chunk is relevant. Both vector and hybrid search return thousands of individually-relevant chunks but cannot synthesize a thematic overview. **GraphRAG's global community search** map-reduces over Leiden community summaries, producing a coherent taxonomy of research directions.

### Concrete retrieval comparison (from our test corpus)

| Query | Vector (dense-only) | Hybrid (dense+BM25+rerank) | GraphRAG |
|---|---|---|---|
| "How does BM25 rank documents?" | Returns RAG overview chunk (semantic match) | **Returns exact BM25-explanation chunk #1** ✅ | N/A — not relational |
| "Who authored the FLARE paper?" | Returns FLARE paper abstract | Returns FLARE abstract (no improvement) | **Returns full author list with entity links** ✅ |
| "What topics are covered?" | 3 unrelated paper snippets | 3 slightly better snippets | **Returns community summary: retrieval methods, knowledge graphs, agent architectures** ✅ |
| "Show me the retriever implementation" | Returns paper text about retrieval | **Returns `src/retriever.py` from cloned repo** ✅ | N/A |

**The adaptive router chooses the right mode per query** — a small classifier LLM (Qwen 1.5B on `local_gpu`, Claude Haiku on the API profile) decides the path, and the CRAG critic can re-route on low-confidence results.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              DATA SOURCES                        │
                    │   arXiv PDFs  ·  GitHub Repos  ·  Tables/Figures │
                    └──────────────────────┬──────────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────────┐
                    │            INGESTION PIPELINE                    │
                    │  Docling (PDF layout + tables)                   │
                    │  tree-sitter (code chunking)                    │
                    │  Contextual Retrieval (doc-grounding per chunk) │
                    └──────────────────────┬──────────────────────────┘
                                           ▼
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
    ┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
    │  bge-small      │         │  BM25 Sparse     │         │  GraphRAG        │
    │  (384-dim dense)│         │  (fastembed)     │         │  MS GraphRAG →   │
    │                 │         │                  │         │  Kùzu (Cypher)   │
    └────────┬────────┘         └────────┬─────────┘         └────────┬─────────┘
             │                           │                            │
             └──────────┬────────────────┘                            │
                        ▼                                              │
              ┌──────────────────┐                                     │
              │  RRF Fusion (k60)│                                     │
              │  → bge-reranker  │                                     │
              └────────┬─────────┘                                     │
                       │                                               │
                       ▼                                               │
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    LANGGRAPH AGENT (7 nodes)                        │
    │                                                                      │
    │   Router → [Vector │ Hybrid │ Graph │ Tool] → Critic → Synthesizer  │
    │             ↑                            │                          │
    │             └──── retry (CRAG, max 2) ───┘                          │
    │                                                                      │
    │   MCP Tools: search_corpus · read_paper · search_chunks · ...       │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │    Qwen 2.5-1.5B (4-bit)     │
                    │    → cited answer            │
                    └──────────────────────────────┘
```

---

## ✨ Key Features

### 🔍 Hybrid Retrieval (not just vector search)
- **Dense**: bge-small-en-v1.5 (384-dim, 33M params) for semantic similarity
- **Sparse**: BM25 via fastembed for exact keyword matching
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) — combines both result sets
- **Rerank**: bge-reranker-base cross-encoder narrows top-60 → top-8
- Why? Pure vector search misses exact terms (code identifiers, author names, acronyms). BM25 misses semantic matches. Hybrid gets both.

### 🕸️ GraphRAG Layer
- **Microsoft GraphRAG** indexer extracts entities, relationships, and Leiden communities from the corpus
- **Kùzu** (embedded graph DB) stores the result for Cypher-based queries
- **Three search modes**: local (entity neighborhood), global (community summaries), drift (multi-hop traversal)
- Why? "Who cited whom" and "what are the main themes" questions are impossible with vector search alone

### 🤖 Adaptive Multi-Agent (LangGraph)
- **Router** (small-LLM classifier, structured JSON): picks one of 6 routes per query
- **Retriever**: executes the chosen strategy (vector / hybrid / graph / tool)
- **Critic**: CRAG pattern — checks retrieval faithfulness, triggers up to 2 retries
- **Synthesizer**: generates the final answer with inline citations
- **Tool Executor**: dispatches to real MCP tools when needed

### 🔧 Real MCP Integration
- 5 purpose-built tools that operate on the actual downloaded corpus
- Standalone FastMCP server — any MCP client (Claude Desktop, OpenCode, Cursor) can connect
- Tools: `search_corpus`, `read_paper`, `list_papers`, `search_chunks`, `read_repo_file`

### 🏠 100% Local by Default (No API Keys)
- On the default `local_gpu` profile, all models run on your machine via HuggingFace transformers
- bge-small (embeddings) + bge-reranker (reranking) + Qwen 1.5B 4-bit (generation)
- Graceful CPU fallback when no GPU is detected
- Zero cloud dependency, zero rate limits, zero cost — the API profile is opt-in (see [Provider Profiles](#-provider-profiles))

### 📊 Eval + Observability
- Custom LLM-as-judge harness: faithfulness, answer relevancy, context precision/recall
- Pure-Python retrieval metrics: nDCG, MRR, Hit@5
- Langfuse tracing on every agent node (no-op fallback without keys)
- PII redaction before any payload leaves the process

### 🎨 Next.js 16 UI
- Real-time chat interface with markdown rendering
- Agent trace visualization (route badge, retry count, source count)
- Expandable citation cards with doc_id, page, chunk_id
- Architecture page with interactive pipeline diagram
- Dark theme, responsive, health indicator

---

## 🚀 Quick Start

```bash
git clone https://github.com/bhavya998/hybrid-graph-agent.git
cd hybrid-graph-agent

# Backend
uv sync                           # install Python deps
hga fetch                         # download arXiv papers + repos (no GPU needed)
hga parse                         # parse → chunks.parquet (no GPU needed)
hga ingest                        # embed + index into Qdrant (downloads models)
hga graph                         # build knowledge graph (downloads Qwen)
hga serve                         # FastAPI on :8000

# Frontend (separate terminal)
cd ui && npm install && npm run dev   # Next.js on :3000
```

Open `http://localhost:3000` and start asking questions.

> Models auto-download from HuggingFace on first use (~3GB total). Cached thereafter.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dim, 33M params, MTEB-top for size |
| **Sparse Retrieval** | `fastembed` (Qdrant/bm25) | Local BM25, no server |
| **Reranker** | `BAAI/bge-reranker-base` | 280M cross-encoder, strong on MS-MARCO |
| **Vector Store** | Qdrant (local file mode) | Hybrid native (dense+sparse+RRF in one query) |
| **Graph DB** | Kùzu (embedded) | Cypher-compatible, no Docker, no Java |
| **GraphRAG** | Microsoft GraphRAG | Leiden communities, canonical extraction pipeline |
| **LLM** | `Qwen2.5-1.5B-Instruct` (4-bit) | Fits in ~2GB VRAM, strong for size |
| **Agent Orchestration** | LangGraph 1.x | Stateful, checkpointed, adaptive routing |
| **MCP** | FastMCP + langchain-mcp-adapters | Real tool server, any client can connect |
| **Backend** | FastAPI + Uvicorn | Async, typed, OpenAPI |
| **Frontend** | Next.js 16 + React 19 + Tailwind 4 | App Router, Turbopack |
| **PDF Parsing** | Docling | Layout-aware, table extraction, figure detection |
| **Code Parsing** | tree-sitter (7 languages) | AST-aware symbol-level chunking |
| **Tracing** | Langfuse (with no-op fallback) | OTel-native, vendor-neutral |
| **Eval** | Custom LLM-as-judge harness | Faithfulness, nDCG, MRR — no fragile deps |

---

## 📁 Project Structure

```
hybrid-graph-agent/
├── src/hga/
│   ├── config/              Settings + provider-profile abstraction
│   ├── chunking/            Chunk schema, semantic chunker, Contextual Retrieval
│   ├── ingestion/           arXiv/repo fetchers, Docling PDF handler, tree-sitter handler
│   ├── embeddings/          Embedder interface + bge-small (local) + Voyage (API)
│   ├── retrieval/           Qdrant hybrid store + bge-reranker + Cohere (API)
│   ├── graph/               GraphRAG indexer + Kùzu store + local/global/drift search
│   ├── agents/              LangGraph: router, retriever, critic, synthesizer
│   ├── tools/               MCP corpus server (5 real tools)
│   ├── llm/                 LLM interface + Qwen (local) + Anthropic (API)
│   ├── eval/                LLM-as-judge eval harness
│   ├── observability/       Langfuse tracing + PII redaction
│   └── serving/             FastAPI + Chainlit
├── ui/                      Next.js 16 frontend (chat + architecture page)
├── tests/                   12 tests (6 hybrid store + 6 observability)
├── eval/golden/             Golden eval set (JSONL)
├── docs/
│   └── MIGRATION_TO_DATABRICKS.md
└── AGENTS.md                Coding-agent contract
```

---

## 🖥️ CLI Commands

| Command | Description | Needs GPU? |
|---|---|---|
| `hga fetch` | Download arXiv papers + clone repos | No |
| `hga parse` | Parse PDFs/repos → `chunks.parquet` | No |
| `hga ingest` | Full pipeline: embed + index into Qdrant | Yes |
| `hga graph` | GraphRAG extraction → Kùzu graph | Yes |
| `hga serve` | FastAPI server on `:8000` | Yes |
| `hga eval` | Run eval harness against golden set | Yes |

---

## 🔌 API Endpoints

```bash
# Health check
GET /health → {"status": "ok"}

# Ask a question
POST /ask
  Body: {"query": "What is FLARE?"}
  Response: {
    "answer": "FLARE is Forward-Looking Active Retrieval...",
    "citations": [{"chunk_id": "...", "doc_id": "2305.06983", "page": null, "score": 0.95}],
    "route": "hybrid",
    "retry_count": 0
  }
```

---

## 🧪 Testing

```bash
uv run pytest -v              # 12 tests: 6 hybrid store + 6 observability
uv run ruff check src tests   # lint (clean)
uv run mypy src               # typecheck strict (49 files, clean)
```

---

## 🔄 Provider Profiles

The system supports swappable backends via a single config flag:

| Profile | LLM | Embeddings | Reranker | When to use |
|---|---|---|---|---|
| `local_gpu` (default) | Qwen 1.5B 4-bit | bge-small | bge-reranker | No API keys, full privacy |
| `local` | Claude Sonnet/Haiku | Voyage-3-large | Cohere | Best quality, costs $ |
| `databricks` | Foundation Model APIs | Vector Search | AI Gateway | Enterprise deployment |

Switch: `HGA_PROFILE=local` in `.env`

See [`docs/MIGRATION_TO_DATABRICKS.md`](docs/MIGRATION_TO_DATABRICKS.md) for the full Databricks/Mosaic AI migration map.

---

## 🗺️ Databricks / Mosaic AI Parity

Every component maps 1:1 to a Databricks equivalent. The provider-profile abstraction means the swap is config-driven — no agent or ingestion logic changes.

| Local | Databricks |
|---|---|
| Qwen (local) | Foundation Model APIs via AI Gateway |
| bge-small (local) | Foundation Model embedding endpoint |
| bge-reranker (local) | Foundation Model rerank |
| Qdrant (local) | Vector Search (hybrid mode) |
| Kùzu (local) | Delta + GraphFrames |
| FastAPI | Model Serving (Agent endpoint) |
| Langfuse | MLflow Tracing |
| Custom eval | `databricks.agents.eval` |
| chunks.parquet | Unity Catalog Delta table |

---

## 📈 What Was Tested Live

Every component has been verified with real model downloads and real data — no mocks, no stubs:

| Component | Verified With |
|---|---|
| bge-small embeddings | 5 documents → 384-dim vectors |
| bge-reranker | Query/doc pairs scored correctly |
| Qwen 1.5B (text) | Generated coherent RAG definition |
| Qwen 1.5B (JSON) | Produced valid structured JSON |
| Qdrant hybrid + RRF | BM25 doc ranked #1 for BM25 query |
| MCP tools (5) | All return real corpus data |
| LangGraph | 7-node graph compiles at runtime |
| PDF parsing (Docling) | 91 chunks from real arXiv PDF |
| Code parsing (tree-sitter) | 49 chunks from FLARE repo |
| CPU fallback | Works gracefully without CUDA |

---

## 📄 License

MIT

---

<div align="center">

Built as a portfolio project demonstrating a hybrid RAG + GraphRAG + agent architecture with modern (2026) practices.

</div>
