"""hga command-line entrypoint."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from hga.ingestion.pipeline import IngestPipeline

app = typer.Typer(help="Hybrid RAG + GraphRAG + Multi-Agent.", no_args_is_help=True)


def _utf8_stdout() -> None:
    """Force UTF-8 stdout/stderr so unicode chunk content prints on Windows."""
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8")


def _pipeline() -> IngestPipeline:
    """Lazy import so CLI start-up stays fast."""
    from hga.ingestion.pipeline import IngestPipeline

    return IngestPipeline()


@app.command()
def ingest(
    max_papers: int = typer.Option(40, help="Max arXiv papers to fetch"),
    max_repos: int = typer.Option(12, help="Max GitHub repos to clone"),
    no_fetch: bool = typer.Option(False, help="Skip fetch stage (reuse data/raw)"),
    no_embed: bool = typer.Option(False, help="Stop after parse (no API keys needed)"),
) -> None:
    """Fetch + parse + contextualize + embed + index the corpus."""
    _utf8_stdout()
    from hga.ingestion.pipeline import IngestPipeline

    pipe = IngestPipeline()
    asyncio.run(
        pipe.run(
            max_papers=max_papers,
            max_repos=max_repos,
            skip_fetch=no_fetch,
            embed=not no_embed,
        )
    )


@app.command()
def fetch(
    max_papers: int = typer.Option(40),
    max_repos: int = typer.Option(12),
) -> None:
    """Fetch arXiv papers + linked repos only (no parsing, no API)."""
    _utf8_stdout()
    from hga.ingestion.pipeline import IngestPipeline

    IngestPipeline().fetch(max_papers=max_papers, max_repos=max_repos)


@app.command()
def parse() -> None:
    """Parse already-fetched data/raw/ into chunks.parquet (no API keys)."""
    _utf8_stdout()
    from hga.ingestion.pipeline import IngestPipeline

    IngestPipeline().parse()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Run the FastAPI server."""
    _utf8_stdout()
    import uvicorn

    uvicorn.run("hga.serving.api:app", host=host, port=port, reload=True)


@app.command()
def graph() -> None:
    """Run Microsoft GraphRAG indexer + load entities/relations into Kùzu."""
    _utf8_stdout()
    from hga.config import get_settings
    from hga.graph.graphrag_indexer import GraphragIndexer
    from hga.graph.kuzu_store import KuzuGraphStore

    s = get_settings()
    pipeline = _pipeline()
    chunks = pipeline.load_chunks()
    if not chunks:
        typer.echo("No chunks found. Run 'hga parse' first.")
        raise typer.Exit(1)

    rows = [
        {"doc_id": c.doc_id, "title": c.metadata.get("title", ""), "content": c.content}
        for c in chunks
    ]
    typer.echo(f"GraphRAG indexing {len(rows)} chunks ...")
    indexer = GraphragIndexer(s.graphrag_output)
    indexer.prepare_input(rows)
    indexer.write_settings()
    paths = asyncio.run(indexer.run())
    typer.echo("Index complete. Loading into Kùzu ...")
    store = KuzuGraphStore(path=str(s.kuzu_path))
    asyncio.run(
        store.load(
            str(paths["entities"]), str(paths["relationships"]),
            str(paths["communities"]), str(paths["community_reports"]),
        )
    )
    typer.echo("GraphRAG index + Kùzu load complete.")


@app.command()
def eval(
    golden: str = typer.Option("eval/golden/golden.jsonl", help="Path to golden JSONL"),
    report_dir: str = typer.Option("eval/reports", help="Output directory for reports"),
) -> None:
    """Run LLM-as-judge + retrieval eval harness against the golden set."""
    _utf8_stdout()
    from hga.eval.harness import EvalHarness

    harness = EvalHarness()
    result = harness.run(golden_path=golden, report_dir=report_dir)
    if "error" in result:
        typer.echo(f"Error: {result['error']}")
        return
    typer.echo(f"Eval complete: {result.get('n_items', 0)} items")
    f = result.get("macro_faithfulness", 0)
    r = result.get("macro_relevancy", 0)
    cp = result.get("macro_context_precision", 0)
    cr = result.get("macro_context_recall", 0)
    typer.echo(f"  faithfulness:      {f:.3f}")
    typer.echo(f"  answer relevancy:  {r:.3f}")
    typer.echo(f"  context precision: {cp:.3f}")
    typer.echo(f"  context recall:    {cr:.3f}")
    typer.echo(f"  nDCG:              {result.get('macro_ndcg', 0):.3f}")
    typer.echo(f"  MRR:               {result.get('macro_mrr', 0):.3f}")
    typer.echo(f"  Hit@5:             {result.get('macro_hit_rate_5', 0):.3f}")
    typer.echo(f"Report: {report_dir}/latest.json")


if __name__ == "__main__":
    app()
