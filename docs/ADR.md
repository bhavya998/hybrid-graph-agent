# ADR: Architecture Decision Record — hga

## Status
Accepted

## Context
Building a production-grade research assistant that answers questions across arXiv papers and their linked GitHub repos. The system must handle factual lookups, multi-hop relationship queries, global summarization, and code inspection — all from one interface.

## Decisions

### D1: Hybrid Retrieval (Dense + BM25 + RRF + Cross-encoder Rerank)
**Decision:** Use Qdrant hybrid mode (bge-small dense + BM25 sparse), fuse via Reciprocal Rank Fusion (k=60), then rerank top-60 → top-8 with bge-reranker-base.

**Rationale:** Pure vector search misses exact identifiers (code paths, author names, acronyms). Pure BM25 misses semantic paraphrasing. Empirically, BM25 catches `src/retriever.py` that dense embeddings miss, while dense catches "augmented generation pipeline" that BM25 misses. RRF fuses both without needing score calibration.

**Alternatives considered:**
- Dense-only: rejected after testing — misses exact code identifiers
- BM25-only: rejected — misses semantic queries like "how does X work"
- Cohere Rerank (cloud): adds latency + cost + API dependency for marginal quality gain over local bge-reranker

**Tradeoffs:** Two indexes instead of one (2x storage). Rerank adds ~100ms latency per query. Both acceptable for a research assistant (not real-time).

### D2: GraphRAG Layer (Microsoft GraphRAG → Kùzu)
**Decision:** Run MS GraphRAG indexer for entity extraction + Leiden community detection, store results in Kùzu (embedded graph DB) for Cypher queries.

**Rationale:** "Who cited whom" and "what are the main themes across 40 papers" are impossible with vector search alone. GraphRAG's community summaries enable global map-reduce queries. Kùzu is Cypher-compatible (familiar syntax), embedded (no Docker/Java like Neo4j), and fast for sub-graph traversal.

**Alternatives considered:**
- Neo4j: requires Docker + Java runtime, heavier deployment
- LightRAG: less mature, smaller community
- nano-graphrag: too minimal for production use

**Tradeoffs:** GraphRAG indexing is expensive (LLM calls for entity extraction). Kùzu has a smaller ecosystem than Neo4j.

### D3: Adaptive Multi-Agent (LangGraph)
**Decision:** LangGraph state machine with 7 nodes: Router → [Vector | Hybrid | Graph | Tool] → Critic → Synthesizer, with CRAG retry loop (max 2).

**Rationale:** Different query types need different retrieval strategies. A cheap classifier (Haiku-speed) routes each query, avoiding the cost of always running all three retrieval modes. The CRAG critic catches low-confidence retrievals and escalates to graph search.

**Alternatives considered:**
- Always run all 3 modes and merge: 3x cost, 3x latency
- Single RAG pipeline: can't handle "what are the main themes" (needs graph)
- CrewAI: role-based crews are less flexible than graph-based state machines for conditional routing

**Tradeoffs:** Router misclassification routes to wrong strategy (mitigated by CRAG critic retry). LangGraph has steeper learning curve than simpler frameworks.

### D4: Provider Profiles (Config-Driven Swap)
**Decision:** Single `profile` setting in `Settings` swaps LLM, embeddings, reranker, and vector store implementations without touching agent/graph code.

**Rationale:** Same system runs locally (bge + Qwen, no API keys) or on Databricks (Foundation Model APIs, Vector Search) by changing one env var. This demonstrates architecture awareness — the agent logic is decoupled from infrastructure.

### D5: All-API Stack for `local` Profile
**Decision:** The `local` profile uses Anthropic (LLM), Voyage (embeddings), Cohere (rerank), Langfuse Cloud (tracing). No self-hosted models.

**Rationale:** API providers give best quality per dollar. Self-hosting models adds ops burden (GPU management, model serving, updates). For a portfolio project demonstrating architecture (not DevOps), APIs are the right choice.

## Consequences
- System handles 5 distinct query types with appropriate retrieval strategies
- Config-driven provider swap proves enterprise-readiness (Databricks migration is config-only)
- Hybrid + Graph + Agent = 3x complexity vs simple RAG, but 3x capability
- All interfaces are behind Protocols/ABCs, enabling future provider additions
