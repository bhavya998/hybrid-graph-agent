"""Kùzu graph store — embedded Cypher-graph, loaded from GraphRAG parquets.

Kùzu is a pure-Python embedded graph DB (no server, no Docker). It loads
entities / relationships / communities from the Microsoft GraphRAG indexer
output and exposes local, global, and drift search over Cypher.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import kuzu

from hga.config import get_settings
from hga.graph.store import (
    CommunityContext,
    EntityResult,
    GraphSearchResult,
    GraphStore,
    RelationshipResult,
)


def _safe_str(val: Any) -> str:
    return str(val).strip() if val is not None else ""


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


class KuzuGraphStore(GraphStore):
    """Embedded Kùzu graph loaded from GraphRAG parquet outputs."""

    _TABLES: list[str] = ["Entity", "Relationship", "Community"]

    def __init__(self, path: str | None = None) -> None:
        p = Path(path or get_settings().kuzu_path)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        self._db = kuzu.Database(str(p))
        self._conn = kuzu.Connection(self._db)
        self._loaded = False

    @property
    def name(self) -> str:
        return "kuzu"

    def _run(self, query: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Execute a Cypher query and materialise rows eagerly."""
        result = self._conn.execute(query, params or {})
        return [list(row) for row in result]

    async def load(
        self,
        entities_path: str,
        relationships_path: str,
        communities_path: str,
        reports_path: str,
    ) -> None:

        def _do() -> None:
            self._create_tables()
            self._import_entities(entities_path)
            self._import_relationships(relationships_path)
            self._import_communities(communities_path, reports_path)
            self._loaded = True

        await asyncio.to_thread(_do)

    def _create_tables(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE Entity("
            "  id STRING, name STRING, entity_type STRING,"
            "  description STRING, degree INT64,"
            "  PRIMARY KEY(id)"
            ")"
        )
        self._conn.execute(
            "CREATE REL TABLE RelatedTo(FROM Entity TO Entity,"
            "  description STRING, weight DOUBLE, id STRING)"
        )
        self._conn.execute(
            "CREATE NODE TABLE Community("
            "  id STRING, title STRING, summary STRING,"
            "  level INT64, rating DOUBLE, weight DOUBLE,"
            "  full_content STRING,"
            "  PRIMARY KEY(id)"
            ")"
        )
        self._conn.execute(
            "CREATE REL TABLE BelongsTo(FROM Entity TO Community, weight DOUBLE)"
        )

    def _import_entities(self, path: str) -> None:
        import pandas as pd

        df = pd.read_parquet(path).fillna("")
        for col in ("name", "type", "description"):
            if col not in df.columns:
                df[col] = ""
        for col in ("degree", "human_readable_id"):
            if col not in df.columns:
                df[col] = 0
        name_col = df.columns[0] if "title" not in df.columns else "title"
        if "name" not in df.columns:
            df["name"] = df[name_col]
        df["name"] = df["name"].astype(str).str.strip()
        df["entity_type"] = df.get("type", "").astype(str).str.strip()
        df["description"] = df.get("description", "").astype(str).str.strip()
        df["degree"] = _safe_int(df.get("degree", 0))
        for _, row in df.iterrows():
            raw_id = str(row.get("id", row["name"]))
            eid = raw_id.replace("'", "").replace('"', "")
            self._conn.execute(
                "CREATE (e:Entity {id: $id, name: $name, entity_type: $type,"
                " description: $desc, degree: $deg})",
                {
                    "id": eid,
                    "name": str(row["name"])[:500],
                    "type": str(row["entity_type"])[:200],
                    "desc": str(row["description"])[:2000],
                    "deg": _safe_int(row.get("degree", 0)),
                },
            )

    def _import_relationships(self, path: str) -> None:
        import pandas as pd

        df = pd.read_parquet(path).fillna("")
        for _, row in df.iterrows():
            src = str(row.get("source", "")).replace("'", "").replace('"', "").strip()
            tgt = str(row.get("target", "")).replace("'", "").replace('"', "").strip()
            if not src or not tgt:
                continue
            desc = _safe_str(row.get("description", ""))[:2000]
            weight = _safe_float(row.get("weight", 1.0))
            rel_id = _safe_str(row.get("id", f"{src}-{tgt}"))
            self._conn.execute(
                "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})"
                " CREATE (a)-[:RelatedTo {description: $desc, weight: $w, id: $id}]->(b)",
                {"src": src, "tgt": tgt, "desc": desc, "w": weight, "id": rel_id},
            )

    def _import_communities(self, communities_path: str, reports_path: str) -> None:
        import pandas as pd

        comm = pd.read_parquet(communities_path).fillna("")
        reports = pd.read_parquet(reports_path).fillna("")
        rep_map: dict[str, str] = {}
        if "community" in reports.columns and "full_content" in reports.columns:
            for _, rrow in reports.iterrows():
                rep_map[str(rrow["community"])] = str(rrow["full_content"])
        for _, row in comm.iterrows():
            cid = str(row.get("community", ""))
            title = _safe_str(row.get("title", ""))[:500]
            summary = _safe_str(row.get("summary", ""))[:5000]
            level = _safe_int(row.get("level", 0))
            rating = _safe_float(row.get("rating", 0))
            weight = _safe_float(row.get("weight", 1))
            full = rep_map.get(cid, summary)
            try:
                self._conn.execute(
                    "CREATE (c:Community {id: $id, title: $title, summary: $summary,"
                    " level: $level, rating: $rating, weight: $weight, full_content: $full})",
                    {"id": cid, "title": title, "summary": summary,
                     "level": level, "rating": rating, "weight": weight, "full": full},
                )
            except Exception:
                continue

    async def local_search(self, query: str, top_k: int = 8) -> GraphSearchResult:
        if not self._loaded:
            return GraphSearchResult(query=query, answer="Graph not loaded.")
        entities = await self._entity_match(query, top_k)
        relationships = await self._expand_relationships(entities, top_k)
        answer = f"Found {len(entities)} entities matching '{query}'."
        return GraphSearchResult(
            query=query,
            answer=answer,
            entities=entities,
            relationships=relationships,
        )

    async def global_search(self, query: str, community_level: int = 0) -> GraphSearchResult:
        if not self._loaded:
            return GraphSearchResult(query=query, answer="Graph not loaded.")
        communities = await self._community_match(community_level, top_k=5)
        summaries = [c.summary for c in communities if c.summary]
        answer = (
            "\n\n".join(summaries)
            if summaries
            else f"No communities at level {community_level}."
        )
        return GraphSearchResult(
            query=query,
            answer=answer,
            communities=communities,
        )

    async def drift_search(self, query: str, steps: int = 3) -> GraphSearchResult:
        if not self._loaded:
            return GraphSearchResult(query=query, answer="Graph not loaded.")
        seeds = await self._entity_match(query, top_k=3)
        if not seeds:
            return GraphSearchResult(query=query, answer="No seed entities found.")
        all_entities = list(seeds)
        all_rels: list[RelationshipResult] = []
        visited: set[str] = set()
        frontier = seeds
        for _step in range(steps):
            if not frontier:
                break
            new_frontier: list[EntityResult] = []
            for ent in frontier:
                expanded = await self._expand_relationships([ent], top_k=5)
                all_rels.extend(expanded)
                for rel in expanded:
                    if rel.target not in visited:
                        visited.add(rel.target)
                        new_frontier.append(
                            EntityResult(
                                name=rel.target,
                                entity_type="",
                                description=rel.description[:200],
                                degree=0,
                            )
                        )
            all_entities.extend(new_frontier)
            frontier = new_frontier
        return GraphSearchResult(
            query=query,
            answer=(
                f"Drift ({steps} steps, {len(all_entities)} entities,"
                f" {len(all_rels)} relationships)."
            ),
            entities=all_entities,
            relationships=all_rels,
        )

    async def _entity_match(self, query: str, top_k: int) -> list[EntityResult]:
        def _do() -> list[EntityResult]:
            q = query.replace("'", "").replace('"', "")
            try:
                r = self._run(
                    "MATCH (e:Entity) WHERE CONTAINS(TOLOWER(e.name), TOLOWER($q))"
                    " OR CONTAINS(TOLOWER(e.description), TOLOWER($q))"
                    " RETURN e.name, e.entity_type, e.description, e.degree LIMIT $k",
                    {"q": q, "k": top_k},
                )
                return [
                    EntityResult(name=_safe_str(row[0]), entity_type=_safe_str(row[1]),
                                 description=_safe_str(row[2]), degree=_safe_int(row[3]))
                    for row in r
                ]
            except Exception:
                return []

        return await asyncio.to_thread(_do)

    async def _expand_relationships(
        self, entities: list[EntityResult], top_k: int
    ) -> list[RelationshipResult]:
        if not entities:
            return []

        def _do(name_single: str) -> list[RelationshipResult]:
            try:
                r = self._run(
                    "MATCH (a:Entity {name: $name})-[rel:RelatedTo]->(b:Entity)"
                    " RETURN a.name, b.name, rel.description, rel.weight LIMIT $k",
                    {"name": name_single, "k": top_k},
                )
                return [
                    RelationshipResult(source=_safe_str(row[0]), target=_safe_str(row[1]),
                                       description=_safe_str(row[2]))
                    for row in r
                ]
            except Exception:
                return []

        results: list[RelationshipResult] = []
        for ent in entities:
            results.extend(await asyncio.to_thread(_do, ent.name))
        return results

    async def _community_match(
        self, level: int, top_k: int
    ) -> list[CommunityContext]:
        def _do() -> list[CommunityContext]:
            try:
                r = self._run(
                    "MATCH (c:Community) WHERE c.level = $level"
                    " RETURN c.title, c.level, c.full_content, c.weight"
                    " ORDER BY c.weight DESC LIMIT $k",
                    {"level": level, "k": top_k},
                )
                return [
                    CommunityContext(title=_safe_str(row[0]), level=_safe_int(row[1]),
                                     summary=_safe_str(row[2]), weight=_safe_float(row[3]))
                    for row in r
                ]
            except Exception:
                return []

        return await asyncio.to_thread(_do)
