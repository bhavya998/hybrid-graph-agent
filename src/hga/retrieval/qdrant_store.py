"""Qdrant hybrid vector store — dense (Voyage) + BM25 sparse, RRF fusion.

Uses the path-mode (local-file) client, which is synchronous. All public
methods are async and delegate via ``asyncio.to_thread`` so the store composes
with the async agent/embedder stack. For a server deployment, swap to
``AsyncQdrantClient(host=...)`` keeping the same models API.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from hga.retrieval.sparse import FastembedBM25, SparseEmbedder
from hga.retrieval.store import SearchResult, VectorRecord, VectorStore

_DENSE = "dense"
_SPARSE = "text"


def _point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id from chunk_id (uuid5 for stable re-upsert)."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _to_filter(filters: dict[str, Any] | None) -> qm.Filter | None:
    """Build an equality filter from a flat {field: value} dict."""
    if not filters:
        return None
    must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filters.items()]
    return qm.Filter(must=must)


def _to_result(point: Any) -> SearchResult:
    payload = dict(point.payload or {})
    return SearchResult(
        chunk_id=str(payload.get("chunk_id", point.id)),
        score=float(point.score),
        payload=payload,
    )


class QdrantStore(VectorStore):
    """Local Qdrant store with hybrid dense+sparse retrieval and RRF fusion."""

    def __init__(
        self,
        path: str,
        collection: str = "chunks",
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._collection = collection
        self._client = QdrantClient(path=path)
        self._sparse = sparse_embedder or FastembedBM25()
        import atexit

        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._client.close()

    @property
    def collection(self) -> str:
        return self._collection

    def close(self) -> None:
        """Release the local store lock. Call on shutdown to avoid teardown noise."""
        self._client.close()

    async def ensure_collection(self, dense_dim: int) -> None:
        def _do() -> None:
            existing = {c.name for c in self._client.get_collections().collections}
            if self._collection in existing:
                info = self._client.get_collection(self._collection)
                vecs = info.config.params.vectors
                current_dim = 0
                if isinstance(vecs, dict) and _DENSE in vecs:
                    current_dim = vecs[_DENSE].size or 0
                elif getattr(vecs, "size", None):
                    current_dim = getattr(vecs, "size", 0)
                if current_dim != dense_dim:
                    self._client.delete_collection(self._collection)
                    existing.discard(self._collection)
                else:
                    return
            self._client.create_collection(
                self._collection,
                vectors_config={
                    _DENSE: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)
                },
                sparse_vectors_config={_SPARSE: qm.SparseVectorParams()},
            )

        await asyncio.to_thread(_do)

    async def upsert(self, records: list[VectorRecord], batch_size: int = 100) -> None:
        contents = [r.content for r in records]
        sparses = await asyncio.to_thread(self._sparse.embed_documents, contents)

        points: list[qm.PointStruct] = []
        for rec, sp in zip(records, sparses, strict=True):
            payload = {
                "chunk_id": rec.chunk_id,
                "doc_id": rec.doc_id,
                "content": rec.content,
                **rec.payload,
            }
            points.append(
                qm.PointStruct(
                    id=_point_id(rec.chunk_id),
                    vector={
                        _DENSE: rec.dense_vector,
                        _SPARSE: qm.SparseVector(indices=sp.indices, values=sp.values),
                    },
                    payload=payload,
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await asyncio.to_thread(self._client.upsert, self._collection, batch)

    async def search_dense(
        self, query: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        def _do() -> list[SearchResult]:
            res = self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=_DENSE,
                query_filter=_to_filter(filters),
                limit=k,
                with_payload=True,
            )
            return [_to_result(p) for p in res.points]

        return await asyncio.to_thread(_do)

    async def search_sparse(
        self, query_text: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        sp = await asyncio.to_thread(self._sparse.embed_query, query_text)
        sparse_vec = qm.SparseVector(indices=sp.indices, values=sp.values)

        def _do() -> list[SearchResult]:
            res = self._client.query_points(
                collection_name=self._collection,
                query=sparse_vec,
                using=_SPARSE,
                query_filter=_to_filter(filters),
                limit=k,
                with_payload=True,
            )
            return [_to_result(p) for p in res.points]

        return await asyncio.to_thread(_do)

    async def search_hybrid(
        self,
        query: list[float],
        query_text: str,
        k: int,
        rrf_k: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        del rrf_k  # Qdrant's RRF uses a fixed internal constant; k kept for API symmetry.
        sp = await asyncio.to_thread(self._sparse.embed_query, query_text)
        sparse_vec = qm.SparseVector(indices=sp.indices, values=sp.values)
        prefetch_limit = max(k * 3, 20)

        def _do() -> list[SearchResult]:
            res = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qm.Prefetch(query=query, using=_DENSE, limit=prefetch_limit),
                    qm.Prefetch(query=sparse_vec, using=_SPARSE, limit=prefetch_limit),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                query_filter=_to_filter(filters),
                limit=k,
                with_payload=True,
            )
            return [_to_result(p) for p in res.points]

        return await asyncio.to_thread(_do)

    async def delete_by_doc(self, doc_id: str) -> None:
        selector = qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            )
        )
        await asyncio.to_thread(
            lambda: self._client.delete(collection_name=self._collection, points_selector=selector)
        )

    async def count(self) -> int:
        def _do() -> int:
            return int(self._client.count(self._collection, exact=True).count)

        return await asyncio.to_thread(_do)
