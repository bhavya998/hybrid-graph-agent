"""Unified chunk schema — the single contract every ingestion handler emits."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """What kind of content a chunk holds, drives embedding strategy."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    PAGE_IMAGE = "page_image"


class Chunk(BaseModel):
    """A unit of retrieved content. Produced by ingestion, consumed by retrieval.

    Attributes:
        chunk_id: Deterministic id (hash of doc_id + content) for idempotent ingest.
        doc_id: Source document identifier (arxiv id, repo name, etc.).
        source_uri: Original location (URL, file path).
        modality: Content kind — selects the embedder.
        section_path: Hierarchical location, e.g. ["Section 3", "3.2 Method"].
        page: Page number for PDFs, None for code/repos.
        language: ISO code for text/code; "n/a" for images.
        content: The chunk text (markdown). For images, a caption/OCR/summary.
        context_summary: Anthropic Contextual-Retrieval prefix, prepended before
            embedding to ground each chunk in its document context.
        embedding_text: Final string sent to the embedder (context + content).
        metadata: Free-form tags: version, acl, repo_path, symbol_name, etc.
        content_hash: sha256 of content for dedupe.
    """

    chunk_id: str
    doc_id: str
    source_uri: str
    modality: Modality
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    language: str = "en"
    content: str
    context_summary: str = ""
    embedding_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str


class Citation(BaseModel):
    """Provenance for a generated answer span — trace-to-metadata."""

    chunk_id: str
    doc_id: str
    page: int | None = None
    span: str
    score: float
