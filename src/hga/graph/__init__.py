"""Graph package — GraphRAG indexer + Kùzu store + search factory."""

from __future__ import annotations

from hga.config import Settings, get_settings
from hga.graph.kuzu_store import KuzuGraphStore
from hga.graph.store import GraphStore

__all__ = ["GraphStore", "KuzuGraphStore", "get_graph_store"]


def get_graph_store(settings: Settings | None = None) -> GraphStore:
    s = settings or get_settings()
    if s.profile in ("local_gpu", "local"):
        return KuzuGraphStore(path=str(s.kuzu_path))
    raise NotImplementedError(f"GraphStore profile {s.profile!r} not yet implemented")
