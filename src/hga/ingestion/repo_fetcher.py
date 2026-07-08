"""GitHub repo fetcher — clones paper-linked code for code-aware retrieval.

Shallow-clones repos discovered in paper metadata into ``data/raw/repos/`` so
the code handler can chunk them alongside the papers. Failures (private,
renamed, gone) are logged and skipped — they don't break the pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class RepoFetcher:
    """Shallow-clones GitHub repos by ``owner/repo`` slug."""

    def __init__(self, out_dir: Path, max_per_repo_files: int | None = None) -> None:
        self.repos_dir = Path(out_dir) / "repos"
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_repo_files = max_per_repo_files

    def fetch(self, repos: list[str]) -> list[RepoResult]:
        """Clone each repo; return results with local path or error message."""
        results: list[RepoResult] = []
        for repo in repos:
            slug = repo.strip().strip("/").rstrip(")")
            if "/" not in slug or " " in slug:
                continue
            slug = slug.split("?")[0]
            name = slug.replace("/", "__")
            dest = self.repos_dir / name
            if dest.exists() and any(dest.iterdir()):
                results.append(RepoResult(slug=slug, path=str(dest), ok=True))
                continue
            dest.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/{slug}.git"
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                results.append(RepoResult(slug=slug, path=str(dest), ok=True))
            except Exception as e:  # noqa: BLE001
                print(f"  [skip-repo] {slug}: {type(e).__name__}")
                results.append(RepoResult(slug=slug, path="", ok=False, error=str(e)))
        return results


class RepoResult:
    """Outcome of cloning one repo."""

    __slots__ = ("slug", "path", "ok", "error")

    def __init__(self, slug: str, path: str, ok: bool, error: str = "") -> None:
        self.slug = slug
        self.path = path
        self.ok = ok
        self.error = error
