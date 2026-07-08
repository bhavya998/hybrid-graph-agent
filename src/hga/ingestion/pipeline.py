"""End-to-end ingest pipeline: fetch -> parse -> contextualize -> embed -> index.

Stages are separable so the corpus can be built and validated without API keys
(fetch + parse write ``chunks.parquet``), then embedding/indexing runs once keys
are present. Idempotent: content_hash dedupes across runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
from rich.console import Console

from hga.chunking.contextual import Contextualizer
from hga.chunking.schema import Chunk, Modality
from hga.config import Settings, get_settings
from hga.embeddings import get_text_embedder
from hga.ingestion.arxiv_fetcher import ArxivFetcher
from hga.ingestion.code_handler import CodeHandler
from hga.ingestion.pdf_handler import PdfHandler
from hga.ingestion.repo_fetcher import RepoFetcher
from hga.llm import get_llm_client
from hga.retrieval import get_vector_store
from hga.retrieval.store import VectorRecord, VectorStore

if TYPE_CHECKING:
    from hga.embeddings.base import Embedder
    from hga.llm.base import LLMClient

console = Console()


def _chunk_to_row(c: Chunk) -> dict[str, object]:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "source_uri": c.source_uri,
        "modality": c.modality.value,
        "section_path": " / ".join(c.section_path),
        "page": c.page,
        "language": c.language,
        "content": c.content,
        "context_summary": c.context_summary,
        "embedding_text": c.embedding_text or c.content,
        "metadata": c.metadata,
        "content_hash": c.content_hash,
    }


def _row_to_chunk(row: pd.Series) -> Chunk:
    return Chunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        source_uri=str(row["source_uri"]),
        modality=Modality(str(row["modality"])),
        section_path=(
            str(row.get("section_path", "")).split(" / ")
            if row.get("section_path")
            else []
        ),
        page=row["page"] if pd.notna(row.get("page")) else None,
        language=str(row.get("language", "en") or "en"),
        content=str(row["content"]),
        context_summary=str(row.get("context_summary", "") or ""),
        embedding_text=str(row.get("embedding_text") or row["content"]),
        metadata=dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {},
        content_hash=str(row["content_hash"]),
    )


class IngestPipeline:
    """Orchestrates the full ingestion path for the arXiv + code corpus."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.s = settings or get_settings()
        self.paths = self.s.paths()
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._pdf = PdfHandler()
        self._code = CodeHandler()

    def fetch(self, max_papers: int = 40, max_repos: int = 12) -> None:
        """Stage 1: fetch arXiv PDFs + clone linked repos. Network only, no API keys."""
        console.print(f"[bold cyan]Stage 1[/] fetching up to {max_papers} papers ...")
        fetcher = ArxivFetcher(self.s.data_dir / "raw", max_per_query=max(8, max_papers // 4))
        papers = fetcher.fetch(max_total=max_papers)
        console.print(f"  fetched {len(papers)} papers -> data/raw/papers/")

        all_repos: list[str] = sorted({r for p in papers for r in p.github_repos})[:max_repos]
        if all_repos:
            console.print(f"[bold cyan]Stage 1b[/] cloning {len(all_repos)} repos ...")
            results = RepoFetcher(self.s.data_dir / "raw").fetch(all_repos)
            ok = sum(1 for r in results if r.ok)
            console.print(f"  cloned {ok}/{len(all_repos)} repos -> data/raw/repos/")
        manifest = self.s.data_dir / "raw" / "manifest.json"
        console.print(f"  manifest: {manifest}")

    def parse(self) -> list[Chunk]:
        """Stage 2: parse PDFs + repos into chunks. Saves chunks.parquet. No API keys."""
        console.print("[bold cyan]Stage 2[/] parsing corpus ...")
        chunks: list[Chunk] = []

        papers_dir = self.s.data_dir / "raw" / "papers"
        for pdf in sorted(papers_dir.glob("*.pdf")):
            doc_id = pdf.stem
            console.print(f"  pdf: {pdf.name}")
            try:
                chunks.extend(
                    self._pdf.parse(str(pdf), doc_id=doc_id, source_uri=f"https://arxiv.org/abs/{doc_id}")
                )
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red]skip {pdf.name}: {type(e).__name__}: {e}[/]")

        repos_dir = self.s.data_dir / "raw" / "repos"
        for repo in sorted(repos_dir.iterdir()):
            if not repo.is_dir():
                continue
            slug = repo.name.replace("__", "/")
            console.print(f"  repo: {slug}")
            try:
                chunks.extend(self._code.parse_repo(str(repo), repo_slug=slug))
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red]skip {slug}: {type(e).__name__}: {e}[/]")

        chunks = self._dedupe(chunks)
        df = pd.DataFrame([_chunk_to_row(c) for c in chunks])
        parquet = self.paths["chunks"]
        df.to_parquet(parquet, index=False)
        console.print(f"  [green]{len(chunks)} chunks[/] -> {parquet}")
        return chunks

    def _dedupe(self, chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        out: list[Chunk] = []
        for c in chunks:
            if c.content_hash in seen:
                continue
            seen.add(c.content_hash)
            out.append(c)
        return out

    def load_chunks(self) -> list[Chunk]:
        df = pd.read_parquet(self.paths["chunks"])
        return [_row_to_chunk(row) for _, row in df.iterrows()]

    async def contextualize_all(self, chunks: list[Chunk]) -> list[Chunk]:
        """Stage 3: Contextual Retrieval via Haiku. Groups chunks by doc_id."""
        if not self._llm:
            self._llm = get_llm_client(self.s)
        ctx = Contextualizer(self._llm, model=self.s.extractor_model)
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_doc[c.doc_id].append(c)
        out: list[Chunk] = []
        total_docs = len(by_doc)
        for i, (doc_id, doc_chunks) in enumerate(by_doc.items(), 1):
            doc_text = "\n\n".join(c.content for c in doc_chunks[:40])
            console.print(
                f"  [bold cyan]Stage 3[/] contextualize {doc_id}"
                f" ({i}/{total_docs}, {len(doc_chunks)} chunks)"
            )
            contexts = await ctx.contextualize(doc_text, doc_chunks)
            out.extend(ctx.apply(doc_chunks, contexts))
        return out

    async def embed_and_index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Stage 4: Voyage embed + Qdrant hybrid upsert."""
        if not self._embedder:
            self._embedder = get_text_embedder(self.s)
        if not self._store:
            self._store = get_vector_store(self.s)
        console.print(f"[bold cyan]Stage 4[/] embedding + indexing {len(chunks)} chunks ...")
        await self._store.ensure_collection(self._embedder.dim)

        texts = [c.embedding_text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            console.print(f"  embed batch {i}..{i + len(batch)}")
            vectors.extend(await self._embedder.embed_documents(batch))

        records: list[VectorRecord] = []
        for c, vec in zip(chunks, vectors, strict=True):
            records.append(
                VectorRecord(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    content=c.content,
                    dense_vector=vec,
                    payload={
                        "modality": c.modality.value,
                        "section_path": c.section_path,
                        "language": c.language,
                        "source_uri": c.source_uri,
                        "context_summary": c.context_summary,
                        **c.metadata,
                    },
                )
            )
        await self._store.upsert(records, batch_size=100)
        count = await self._store.count()
        console.print(f"  [green]indexed. total points: {count}[/]")

    async def run(
        self,
        max_papers: int = 40,
        max_repos: int = 12,
        skip_fetch: bool = False,
        embed: bool = True,
    ) -> None:
        """Full pipeline. If embed=False, stops after parse (no API keys needed)."""
        if not skip_fetch:
            self.fetch(max_papers=max_papers, max_repos=max_repos)
        chunks = self.parse()
        if not embed:
            console.print("[yellow]--no-embed: stopping after parse (chunks.parquet written)[/]")
            return
        chunks = await self.contextualize_all(chunks)
        await self.embed_and_index(chunks)
