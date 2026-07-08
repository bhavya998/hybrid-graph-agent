"""Graph store interface + search result types.

The graph layer runs alongside the vector store. It answers relational,
multi-hop, and thematic questions that vector search alone cannot. The
implementation can be anything from Neo4j to Kùzu — the interface stays fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class EntityResult:
    """A matched entity with its description and community context."""

    name: str
    entity_type: str
    description: str
    degree: int = 0


@dataclass(slots=True)
class RelationshipResult:
    """An edge between two entities with its description."""

    source: str
    target: str
    description: str


@dataclass(slots=True)
class CommunityContext:
    """A community summary — one level of the Leiden hierarchy."""

    title: str
    level: int
    summary: str
    weight: float = 0.0


@dataclass(slots=True)
class GraphSearchResult:
    """Aggregated response from a graph search."""

    query: str
    answer: str
    entities: list[EntityResult] = field(default_factory=list)
    relationships: list[RelationshipResult] = field(default_factory=list)
    communities: list[CommunityContext] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GraphStore(Protocol):
    """Graph layer for entity/relationship/community queries."""

    @property
    def name(self) -> str: ...

    async def load(
        self,
        entities_path: str,
        relationships_path: str,
        communities_path: str,
        reports_path: str,
    ) -> None: ...

    async def local_search(self, query: str, top_k: int = 8) -> GraphSearchResult: ...

    async def global_search(self, query: str, community_level: int = 0) -> GraphSearchResult: ...

    async def drift_search(self, query: str, steps: int = 3) -> GraphSearchResult: ...
