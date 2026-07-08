"""Microsoft GraphRAG indexer — entity extraction → Leiden communities.

Prepares input text files from parsed chunks, writes a graphrag settings.yaml
with Anthropic as the LLM provider, then runs the index pipeline. The parquet
outputs are loaded by KuzuGraphStore for Cypher-based querying.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphrag.api.index import build_index
from graphrag.config.load_config import load_config

if TYPE_CHECKING:
    from graphrag.config.models.graph_rag_config import GraphRagConfig


class GraphragIndexer:
    """Wraps the Microsoft GraphRAG indexing pipeline with Anthropic config."""

    def __init__(
        self,
        root_dir: Path,
        anthropic_model: str = "anthropic/claude-haiku-4-5-20251001",
        anthropic_key: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self._model = anthropic_model
        if anthropic_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        self.input_dir = self.root_dir / "input"
        self.output_dir = self.root_dir / "output"

    def prepare_input(self, chunks: list[dict[str, str]], doc_group_field: str = "doc_id") -> None:
        """Write one text file per document into the graphrag input directory."""
        if self.input_dir.exists():
            shutil.rmtree(self.input_dir, ignore_errors=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[str]] = {}
        for c in chunks:
            gid = c.get(doc_group_field, "default")
            groups.setdefault(gid, []).append(c.get("content", ""))
        for gid, texts in groups.items():
            safe = gid.replace("/", "_").replace("\\", "_")
            path = self.input_dir / f"{safe}.txt"
            path.write_text("\n\n---\n\n".join(texts), encoding="utf-8")

    def write_settings(self, **overrides: Any) -> Path:
        """Create a minimal settings.yaml for Anthropic-driven indexing."""
        settings = {
            "root_dir": str(self.root_dir.resolve()),
            "models": {
                "default_chat_model": {
                    "model": self._model,
                    "model_provider": "anthropic",
                    "model_type": "chat",
                    "api_key": "env:ANTHROPIC_API_KEY",
                },
                "default_embedding_model": {
                    "model": "voyage-3-large",
                    "model_provider": "voyage",
                    "model_type": "embedding",
                    "api_key": "env:VOYAGE_API_KEY",
                },
            },
            "input": {
                "type": "text",
                "file_type": "text",
                "base_dir": "input",
                "file_encoding": "utf-8",
            },
            "output": {
                "type": "parquet",
                "base_dir": "output",
            },
            "chunking": {
                "strategy": "tokens",
                "size": 1200,
                "overlap": 100,
            },
            "concurrent_requests": 8,
            **overrides,
        }
        path = self.root_dir / "settings.yaml"
        import yaml
        path.write_text(yaml.safe_dump(settings, default_flow_style=False), encoding="utf-8")
        return path

    async def run(self, method: str = "standard") -> dict[str, Path]:
        """Run the indexing pipeline, returning paths to the output parquets."""
        config: GraphRagConfig = load_config(self.root_dir)
        await build_index(config, method=method)
        return {
            "entities": self.output_dir / "create_final_entities.parquet",
            "relationships": self.output_dir / "create_final_relationships.parquet",
            "communities": self.output_dir / "create_final_communities.parquet",
            "community_reports": self.output_dir / "create_final_community_reports.parquet",
            "text_units": self.output_dir / "create_final_text_units.parquet",
        }
