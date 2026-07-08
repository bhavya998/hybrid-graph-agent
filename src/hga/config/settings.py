"""Application settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Selects concrete implementations via `profile`.

    Profiles:
        local_gpu — self-hosted on GPU (bge-small + bge-reranker-base + Qwen 1.5B 4-bit)
        local     — all-API (Anthropic + Voyage + Cohere)
        databricks — Databricks/Mosaic AI parity (future)

    All secrets from environment / local `.env`. Never hardcode keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HGA_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider profile ---
    profile: Literal["local_gpu", "local", "databricks"] = "local_gpu"

    # --- API models (local profile) ---
    planner_model: str = "claude-sonnet-4-5-20250929"
    extractor_model: str = "claude-haiku-4-5-20251001"
    text_embed_model: str = "voyage-3-large"
    multimodal_embed_model: str = "voyage-multimodal-3"
    rerank_model: str = "rerank-v3.5"

    # --- local GPU models (local_gpu profile) ---
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    local_rerank_model: str = "BAAI/bge-reranker-base"
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_llm_4bit: bool = True

    # --- paths ---
    data_dir: Path = Path("./data")
    qdrant_path: Path = Path("./data/qdrant")
    kuzu_path: Path = Path("./data/kuzu")
    graphrag_output: Path = Path("./data/graphrag_output")
    chunks_path: Path = Path("./data/processed/chunks.parquet")

    # --- retrieval defaults ---
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_k: int = 8
    rrf_k: int = 60

    # --- agent defaults ---
    max_critique_retries: int = 2

    # --- logging ---
    log_level: str = "INFO"

    def paths(self) -> dict[str, Path]:
        """Return all storage paths, ensuring parents exist (chunks=file, rest=dirs)."""
        out = {
            "data": self.data_dir,
            "qdrant": self.qdrant_path,
            "kuzu": self.kuzu_path,
            "graphrag": self.graphrag_output,
            "chunks": self.chunks_path,
        }
        for key, p in out.items():
            if key == "chunks":
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
        return out


def get_settings() -> Settings:
    """Return a cached Settings instance (re-reads env each call in tests)."""
    return Settings()
