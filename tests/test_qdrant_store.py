"""Integration test for the Qdrant hybrid store: upsert + dense/sparse/hybrid search.

Uses real BM25 sparse vectors (fastembed, local) and synthetic dense vectors, so
no API keys are required. Run: uv run pytest tests/test_qdrant_store.py -v
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from hga.retrieval.qdrant_store import QdrantStore
from hga.retrieval.store import VectorRecord


@pytest.fixture()
def store(tmp_path: Path) -> QdrantStore:
    s = QdrantStore(path=str(tmp_path / "qdrant"), collection="test_chunks")
    asyncio.run(s.ensure_collection(dense_dim=4))
    yield s
    s.close()


DOCS = [
    ("c1", "p1", "Retrieval-augmented generation grounds language models in documents."),
    ("c2", "p2", "GraphRAG builds a knowledge graph for multi-hop relational reasoning."),
    ("c3", "p3", "BM25 ranks documents using term frequency and inverse document frequency."),
    ("c4", "p4", "Vector embeddings map text into a high-dimensional similarity space."),
]

# Synthetic dense vectors engineered so c3 is closest to the query vector.
DENSE = {
    "c1": [0.12, 0.85, 0.10, 0.40],
    "c2": [0.80, 0.11, 0.55, 0.20],
    "c3": [0.15, 0.20, 0.10, 0.95],
    "c4": [0.30, 0.30, 0.90, 0.10],
}
QUERY_DENSE = [0.10, 0.15, 0.05, 0.99]  # nearest to c3


@pytest.fixture()
def populated(store: QdrantStore) -> QdrantStore:
    records = [
        VectorRecord(
            chunk_id=cid,
            doc_id=doc,
            content=text,
            dense_vector=DENSE[cid],
            payload={"section": "intro"},
        )
        for cid, doc, text in DOCS
    ]
    asyncio.run(store.upsert(records))
    return store


def test_upsert_count(populated: QdrantStore) -> None:
    assert asyncio.run(populated.count()) == 4


def test_dense_search(populated: QdrantStore) -> None:
    results = asyncio.run(populated.search_dense(QUERY_DENSE, k=2))
    assert len(results) == 2
    assert results[0].chunk_id == "c3"


def test_sparse_search_finds_bm25_doc(populated: QdrantStore) -> None:
    results = asyncio.run(populated.search_sparse("BM25 term frequency document ranking", k=2))
    ids = [r.chunk_id for r in results]
    assert "c3" in ids


def test_hybrid_search(populated: QdrantStore) -> None:
    results = asyncio.run(
        populated.search_hybrid(QUERY_DENSE, "how does BM25 rank documents?", k=3)
    )
    assert len(results) == 3
    # c3 should dominate: it wins both the dense and the sparse side.
    assert results[0].chunk_id == "c3"


def test_filter_by_doc(populated: QdrantStore) -> None:
    results = asyncio.run(populated.search_dense(QUERY_DENSE, k=4, filters={"doc_id": "p3"}))
    assert all(r.payload["doc_id"] == "p3" for r in results)


def test_delete_by_doc(populated: QdrantStore) -> None:
    asyncio.run(populated.delete_by_doc("p3"))
    assert asyncio.run(populated.count()) == 3
