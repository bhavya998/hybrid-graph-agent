"""Smoke test: exercise every Day-1 dependency end-to-end (no API keys needed).

Run: uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def section(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def main() -> int:
    failures: list[str] = []

    section("1. Qdrant local file mode + sparse/hybrid config")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            PointStruct,
            SparseVectorParams,
            VectorParams,
        )

        p = tempfile.mkdtemp()
        try:
            c = QdrantClient(path=p)
            c.create_collection(
                "t",
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
                sparse_vectors_config={"text": SparseVectorParams()},
            )
            c.upsert("t", [PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"t": "hello"})])
            # qdrant-client 1.12+: search() removed, use query_points()
            r = c.query_points("t", query=[0.1, 0.2, 0.3, 0.4], limit=1)
            score = r.points[0].score
            print(f"  OK — score={score:.3f}, sparse vectors supported (BM25-ready)")
        finally:
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        failures.append(f"Qdrant: {type(e).__name__}: {e}")

    section("2. Kùzu embedded graph DB + Cypher")
    try:
        import kuzu

        # Kùzu wants a path it can create; must NOT pre-exist as a directory.
        p = os.path.join(tempfile.gettempdir(), f"kuzu_smoke_{os.getpid()}")
        shutil.rmtree(p, ignore_errors=True)
        try:
            db = kuzu.Database(p)
            conn = kuzu.Connection(db)
            conn.execute("CREATE NODE TABLE Paper(id STRING, title STRING, PRIMARY KEY(id))")
            conn.execute("CREATE REL TABLE CITES(FROM Paper TO Paper)")
            conn.execute("CREATE (p:Paper {id: 'a1', title: 'RAG'})")
            conn.execute("CREATE (p:Paper {id: 'a2', title: 'GraphRAG'})")
            conn.execute(
                "MATCH (a:Paper{id:'a1'}),(b:Paper{id:'a2'}) CREATE (a)-[:CITES]->(b)"
            )
            r = conn.execute("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.title, b.title")
            row = r.get_next()
            print(f"  OK — {row[0]} -> {row[1]}")
        finally:
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        failures.append(f"Kuzu: {type(e).__name__}: {e}")

    section("3. Docling PDF parser")
    try:
        from docling.document_converter import DocumentConverter

        print("  OK — DocumentConverter imported (lazy model load deferred)")
    except Exception as e:  # noqa: BLE001
        failures.append(f"Docling: {type(e).__name__}: {e}")

    section("4. tree-sitter-language-pack")
    try:
        from tree_sitter_language_pack import get_parser

        supported = ["python", "javascript", "typescript", "rust", "go", "cpp", "java"]
        # tree-sitter 0.26: parse() accepts str; root_node() is a method (Node attrs are methods too).
        parsed_ok = 0
        for lang in supported:
            parser = get_parser(lang)
            tree = parser.parse("def f():\n    return 1\n")
            node = tree.root_node()
            if node is not None:
                parsed_ok += 1
        print(f"  OK — {parsed_ok}/{len(supported)} languages parsed to a Node")
    except Exception as e:  # noqa: BLE001
        failures.append(f"tree-sitter: {type(e).__name__}: {e}")

    section("5. arxiv client")
    try:
        import arxiv

        assert hasattr(arxiv, "Client")
        print("  OK — arxiv.Client available")
    except Exception as e:  # noqa: BLE001
        failures.append(f"arxiv: {type(e).__name__}: {e}")

    section("6. Provider clients (no network call)")
    for mod, key_env in [("voyageai", "VOYAGE_API_KEY"), ("cohere", "COHERE_API_KEY"),
                         ("anthropic", "ANTHROPIC_API_KEY")]:
        try:
            __import__(mod)
            present = bool(os.environ.get(key_env))
            print(f"  {mod}: import OK — {key_env}={'set' if present else 'MISSING'}")
            if not present:
                failures.append(f"{key_env} not in env (needed for live calls)")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mod}: {type(e).__name__}: {e}")

    section("7. LangGraph + checkpoint sqlite")
    try:
        from langgraph.graph import StateGraph
        from langgraph.checkpoint.sqlite import SqliteSaver

        print(f"  OK — StateGraph + SqliteSaver available")
    except Exception as e:  # noqa: BLE001
        failures.append(f"langgraph: {type(e).__name__}: {e}")

    section("8. MCP + langchain-mcp-adapters")
    try:
        import mcp
        from langchain_mcp_adapters.client import MultiServerMCPClient

        print("  OK — mcp + MultiServerMCPClient available")
    except Exception as e:  # noqa: BLE001
        failures.append(f"mcp: {type(e).__name__}: {e}")

    section("9. Serving + eval + observability")
    try:
        import fastapi
        import chainlit
        import langfuse
        import mlflow
        import duckdb
        import pandas as pd
        print("  OK — fastapi/chainlit/langfuse/mlflow/duckdb/pandas all import")
    except Exception as e:  # noqa: BLE001
        failures.append(f"serving/eval: {type(e).__name__}: {e}")

    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all Day-1 dependencies operational")
    return 0


if __name__ == "__main__":
    sys.exit(main())
