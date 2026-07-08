"""PDF ingestion handler — Docling → Chunk list (text, tables, figures).

Parses each paper with Docling, splits the markdown export by heading into
sections, then chunks each section. Tables and figures become their own
chunks so the router can prefer them when the question is about results.
"""

from __future__ import annotations

import hashlib
from typing import Any

from hga.chunking.chunker import chunk_text, split_markdown_sections
from hga.chunking.schema import Chunk, Modality


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(doc_id: str, content: str) -> str:
    return hashlib.sha1(f"{doc_id}::{content[:200]}".encode()).hexdigest()[:16]


class PdfHandler:
    """Parses PDFs into text/table/figure Chunks via Docling."""

    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self._conv = DocumentConverter()

    def parse(self, pdf_path: str, doc_id: str, source_uri: str, title: str = "") -> list[Chunk]:
        """Parse one PDF into Chunks. `doc_id` is the arxiv id (or filename)."""
        document = self._conv.convert(pdf_path).document
        chunks: list[Chunk] = []
        md = document.export_to_markdown()

        for section in split_markdown_sections(md):
            for piece in chunk_text(section.text):
                if len(piece) < 60:
                    continue
                chunks.append(self._text_chunk(piece, section.path, doc_id, source_uri, title))

        for i, tbl in enumerate(self._safe_list(getattr(document, "tables", []))):
            md_tbl = self._safe_call(tbl, "export_to_markdown", document)
            if not md_tbl or len(md_tbl) < 20:
                continue
            caption = self._safe_call(tbl, "caption_text")
            content = f"**Table {i + 1}**: {caption}\n\n{md_tbl}" if caption else md_tbl
            chunks.append(
                self._modality_chunk(
                    content,
                    Modality.TABLE,
                    ["Tables"],
                    doc_id,
                    source_uri,
                    title,
                    {"table_index": i},
                )
            )

        for i, pic in enumerate(self._safe_list(getattr(document, "pictures", []))):
            caption = self._safe_call(pic, "caption_text")
            if not caption:
                continue
            content = f"**Figure {i + 1}**: {caption}"
            chunks.append(
                self._modality_chunk(
                    content, Modality.IMAGE, ["Figures"], doc_id, source_uri, title,
                    {"figure_index": i, "needs_image_embedding": True},
                )
            )
        return chunks

    def _text_chunk(
        self, content: str, section_path: list[str], doc_id: str, source_uri: str, title: str
    ) -> Chunk:
        return Chunk(
            chunk_id=_chunk_id(doc_id, content),
            doc_id=doc_id,
            source_uri=source_uri,
            modality=Modality.TEXT,
            section_path=section_path,
            page=None,
            content=content,
            content_hash=_hash(content),
            metadata={"title": title},
        )

    def _modality_chunk(
        self,
        content: str,
        modality: Modality,
        section_path: list[str],
        doc_id: str,
        source_uri: str,
        title: str,
        extra: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            chunk_id=_chunk_id(doc_id, content),
            doc_id=doc_id,
            source_uri=source_uri,
            modality=modality,
            section_path=section_path,
            page=None,
            content=content,
            content_hash=_hash(content),
            metadata={"title": title, **extra},
        )

    @staticmethod
    def _safe_list(val: Any) -> list[Any]:
        return list(val) if val else []

    @staticmethod
    def _safe_call(obj: Any, method: str, *args: Any) -> str:
        fn = getattr(obj, method, None)
        if not callable(fn):
            return ""
        try:
            return str(fn(*args)) or ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _safe_str(val: Any) -> str:
        return str(val).strip() if val else ""
