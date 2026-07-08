"""arXiv paper fetcher — downloads PDFs + metadata for the demo corpus.

Fetches recent AI/ML papers relevant to the demo (RAG, GraphRAG, agents,
reasoning), saves PDFs to ``data/raw/papers/`` and a manifest with metadata +
GitHub links scraped from title/abstract. Idempotent: existing PDFs are reused.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import arxiv

_GH_RE = re.compile(r"github\.com/([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)")


@dataclass(slots=True)
class PaperMeta:
    """Metadata for one fetched paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    pdf_path: str
    github_repos: list[str] = field(default_factory=list)
    published: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _clean_repos(matches: list[str]) -> list[str]:
    out: set[str] = set()
    for m in matches:
        repo = m.strip().rstrip("/").rstrip(").,;")
        if repo.endswith(".git"):
            repo = repo[:-4]
        out.add(repo)
    return sorted(out)


class ArxivFetcher:
    """Fetches arXiv papers by query, deduping by id, caching downloaded PDFs."""

    DEFAULT_QUERIES: list[str] = [
        'ti:"retrieval augmented generation"',
        'abs:"graphrag"',
        'abs:"knowledge graph" abs:"language model"',
        'ti:"tool use" abs:"agent"',
    ]

    def __init__(self, out_dir: Path, max_per_query: int = 12) -> None:
        self.out_dir = Path(out_dir)
        self.papers_dir = self.out_dir / "papers"
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_query = max_per_query
        self._client = arxiv.Client()

    def fetch(
        self,
        queries: list[str] | None = None,
        max_total: int = 40,
    ) -> list[PaperMeta]:
        """Run each query, download PDFs, return deduped metadata. Writes manifest.json."""
        queries = queries or self.DEFAULT_QUERIES
        seen: set[str] = set()
        papers: list[PaperMeta] = []
        for q in queries:
            if len(papers) >= max_total:
                break
            n = min(self.max_per_query, max_total - len(papers))
            search = arxiv.Search(query=q, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
            for result in self._client.results(search):
                aid: str = result.get_short_id()
                base = aid.split("v")[0]
                if base in seen:
                    continue
                seen.add(base)
                meta = self._download(result, aid)
                if meta is not None:
                    papers.append(meta)
        self._write_manifest(papers)
        return papers

    def _download(self, result: Any, aid: str) -> PaperMeta | None:
        safe_id = aid.replace("/", "_")
        pdf_path = self.papers_dir / f"{safe_id}.pdf"
        if not pdf_path.exists():
            pdf_url = result.pdf_url
            if not pdf_url:
                print(f"  [skip] {aid}: no pdf_url")
                return None
            try:
                import requests

                resp = requests.get(pdf_url, timeout=60)
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {aid}: {type(e).__name__}: {e}")
                return None
        abstract = (result.summary or "").replace("\n", " ").strip()
        text = f"{result.title or ''} {abstract}"
        repos = _clean_repos(_GH_RE.findall(text))
        published = ""
        if result.published is not None:
            published = result.published.isoformat()
        return PaperMeta(
            arxiv_id=aid,
            title=(result.title or "").strip(),
            authors=[a.name for a in result.authors],
            abstract=abstract,
            primary_category=result.primary_category or "",
            categories=list(result.categories or []),
            pdf_url=result.pdf_url or "",
            pdf_path=str(pdf_path),
            github_repos=repos,
            published=published,
        )

    def _write_manifest(self, papers: list[PaperMeta]) -> None:
        manifest = self.out_dir / "manifest.json"
        manifest.write_text(
            json.dumps([p.to_dict() for p in papers], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
