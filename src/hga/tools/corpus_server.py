"""Corpus MCP server — real tools over the downloaded arXiv papers + GitHub repos.

Run with:    uv run python -m hga.tools.corpus_server
Or:         mcp run src/hga/tools/corpus_server.py

Exposes 5 tools that operate on the actual downloaded data:
  search_corpus   — full-text search of paper titles / abstracts / authors
  read_paper      — return full metadata for one paper by arxiv id
  list_papers     — list all papers with arxiv id + title
  search_chunks   — keyword search across parsed chunks (from chunks.parquet)
  read_repo_file  — read a file from a cloned GitHub repo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

BASE = Path("./data").resolve()

mcp = FastMCP(
    "hga-corpus",
    instructions="Tools for browsing the downloaded arXiv papers, parsed chunks, and cloned repos.",
)


def _load_manifest() -> list[dict[str, Any]]:
    path = BASE / "raw" / "manifest.json"
    if not path.exists():
        return []
    return cast("list[dict[str, Any]]", json.loads(path.read_text(encoding="utf-8")))


def _load_chunks_df() -> Any:
    import pandas as pd

    path = BASE / "processed" / "chunks.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@mcp.tool()
def search_corpus(query: str) -> str:
    """Search paper titles, abstracts, and authors. Returns matching papers with metadata."""
    papers = _load_manifest()
    if not papers:
        return "No manifest found. Run 'hga fetch' first."
    query_lower = query.lower()
    matches: list[dict[str, Any]] = []
    for p in papers:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        authors = " ".join(p.get("authors", [])).lower()
        if query_lower in title or query_lower in abstract or query_lower in authors:
            matches.append({
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "authors": p.get("authors", [])[:5],
                "categories": p.get("categories", []),
                "published": p.get("published", ""),
            })
    if not matches:
        return f"No papers matching '{query}'. Try broader terms."
    out_lines = [f"{len(matches)} papers match '{query}':"]
    for m in matches[:10]:
        out_lines.append(
            f"  {m['arxiv_id']} — {m['title'][:80]} "
            f"({', '.join(m['authors'][:3])}) [{', '.join(m['categories'][:2])}]"
        )
    if len(matches) > 10:
        out_lines.append(f"  ... and {len(matches) - 10} more")
    return "\n".join(out_lines)


@mcp.tool()
def read_paper(arxiv_id: str) -> str:
    """Return full metadata for a paper: title, all authors, abstract, categories, github repos."""
    papers = _load_manifest()
    for p in papers:
        if p.get("arxiv_id") == arxiv_id:
            return json.dumps({
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "authors": p.get("authors", []),
                "abstract": (p.get("abstract") or "")[:2000],
                "categories": p.get("categories", []),
                "github_repos": p.get("github_repos", []),
                "pdf_url": p.get("pdf_url"),
                "published": p.get("published"),
            }, indent=2, ensure_ascii=False)
    return f"Paper '{arxiv_id}' not found. Use list_papers to see available IDs."


@mcp.tool()
def list_papers() -> str:
    """List all papers in the corpus with arxiv IDs and titles."""
    papers = _load_manifest()
    if not papers:
        return "No papers in corpus. Run 'hga fetch' first."
    lines = [f"{len(papers)} papers in corpus:"]
    for p in papers:
        lines.append(f"  {p['arxiv_id']} — {p.get('title', '?')[:80]}")
    return "\n".join(lines)


@mcp.tool()
def search_chunks(query: str, top_k: int = 5) -> str:
    """Keyword search across chunk content. Returns matching chunk text with doc+section info."""
    try:
        df = _load_chunks_df()
    except Exception:
        return "No chunks.parquet found. Run 'hga parse' first."
    if df.empty:
        return "No chunks in database. Run 'hga parse' first."
    query_lower = query.lower()
    content_col = df["content"].astype(str)
    mask = content_col.str.lower().str.contains(query_lower, na=False)
    matched = df[mask].head(top_k)
    if matched.empty:
        return f"No chunks contain '{query}'. Try different terms."
    results = []
    for _, row in matched.iterrows():
        content = str(row["content"])[:400]
        path = str(row.get("section_path", ""))
        results.append(
            f"  [{row['chunk_id']}] {row['doc_id']} | {path}\n    {content}\n"
        )
    return f"{len(matched)} chunks match '{query}' (showing top {top_k}):\n" + "\n".join(results)


@mcp.tool()
def read_repo_file(repo_slug: str, file_path: str) -> str:
    """Read a file from a cloned GitHub repo.

    repo_slug like 'owner/repo', file_path relative to repo root.
    """
    safe_repo = repo_slug.replace("/", "__")
    repo_dir = BASE / "raw" / "repos" / safe_repo
    if not repo_dir.exists():
        return f"Repo '{repo_slug}' not cloned. Cloned repos: " + ", ".join(
            d.name.replace("__", "/") for d in (BASE / "raw" / "repos").iterdir() if d.is_dir()
        ) if (BASE / "raw" / "repos").exists() else "no repos cloned."
    target = (repo_dir / file_path).resolve()
    if not str(target).startswith(str(repo_dir.resolve())):
        return "Security: path escapes repo root."
    if not target.exists():
        return f"File '{file_path}' not found. Top-level files: " + ", ".join(
            f.name for f in sorted(repo_dir.glob("*.py"))[:20] if f.is_file()
        )
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"
        return content
    except Exception as exc:
        return f"Cannot read file: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
