"""Code ingestion handler — tree-sitter symbol-level chunking.

Walks a repo, parses each code file with the matching tree-sitter grammar, and
emits one CODE chunk per top-level symbol (function/class/struct/...). Top-level
non-symbol statements (imports, constants) accumulate into a module chunk.
Markdown/text files become whole-file TEXT chunks. Skips vendored/build dirs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tree_sitter_language_pack import get_parser

from hga.chunking.schema import Chunk, Modality

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
}
_TEXT_EXT = {".md", ".txt", ".rst"}
_DEF_KINDS = {
    "function_definition", "class_definition", "decorated_definition",
    "function_declaration", "class_declaration", "method_definition",
    "export_statement", "impl_item", "function_item", "struct_item",
    "enum_item", "trait_item", "interface_declaration", "constructor_declaration",
}
_IDENT_KINDS = {"identifier", "type_identifier", "property_identifier", "field_identifier"}
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", ".pytest_cache", ".mypy_cache", "vendor",
}
_MAX_FILE_BYTES = 200_000


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(doc_id: str, content: str) -> str:
    return hashlib.sha1(f"{doc_id}::{content[:200]}".encode()).hexdigest()[:16]


class CodeHandler:
    """Chunks a repo into CODE (and text) Chunks via tree-sitter."""

    def parse_repo(self, repo_path: str, repo_slug: str) -> list[Chunk]:
        """Walk the repo and emit Chunks. `repo_slug` is `owner/repo`."""
        root = Path(repo_path)
        chunks: list[Chunk] = []
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if any(part in _SKIP_DIRS for part in fp.parts):
                continue
            if fp.stat().st_size > _MAX_FILE_BYTES:
                continue
            rel = str(fp.relative_to(root)).replace("\\", "/")
            ext = fp.suffix.lower()
            try:
                src = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if ext in _TEXT_EXT:
                if src.strip():
                    chunks.append(self._text_file_chunk(src, repo_slug, rel))
            elif ext in _EXT_LANG:
                chunks.extend(self._parse_code(src, _EXT_LANG[ext], rel, repo_slug))
        return chunks

    def _parse_code(self, src: str, lang: str, rel_path: str, repo: str) -> list[Chunk]:
        try:
            parser = get_parser(lang)
            tree = parser.parse(src)
        except Exception:  # noqa: BLE001
            return []
        if tree is None:
            return []
        root_node = tree.root_node()
        chunks: list[Chunk] = []
        leftover_start = 0
        for i in range(root_node.named_child_count()):
            child = root_node.named_child(i)
            if child is None:
                continue
            kind = child.kind()
            if kind in _DEF_KINDS:
                start = child.start_byte()
                end = child.end_byte()
                if start > leftover_start + 80:
                    pre = src[leftover_start:start].strip()
                    if pre:
                        chunks.append(self._code_chunk(pre, repo, rel_path, lang, "(module-top)"))
                text = src[start:end]
                name = self._symbol_name(child, src)
                chunks.append(self._code_chunk(text, repo, rel_path, lang, name))
                leftover_start = end
        if leftover_start < len(src):
            tail = src[leftover_start:].strip()
            if tail:
                chunks.append(self._code_chunk(tail, repo, rel_path, lang, "(module-tail)"))
        if not chunks and src.strip():
            chunks.append(self._code_chunk(src, repo, rel_path, lang, "(module)"))
        return chunks

    def _symbol_name(self, node: Any, src: str) -> str:
        for i in range(node.named_child_count()):
            c = node.named_child(i)
            if c is None:
                continue
            if c.kind() in _IDENT_KINDS:
                name = src[c.start_byte():c.end_byte()]
                if name:
                    return str(name)
        return str(node.kind())

    def _code_chunk(self, content: str, repo: str, rel_path: str, lang: str, symbol: str) -> Chunk:
        content = content.strip()
        body = f"# {repo} :: {rel_path} :: {symbol}\n{content}"
        return Chunk(
            chunk_id=_chunk_id(repo, body),
            doc_id=repo,
            source_uri=f"https://github.com/{repo}/blob/HEAD/{rel_path}",
            modality=Modality.CODE,
            section_path=[repo, rel_path],
            page=None,
            language=lang,
            content=body,
            content_hash=_hash(content),
            metadata={"repo": repo, "file": rel_path, "symbol": symbol, "language": lang},
        )

    def _text_file_chunk(self, src: str, repo: str, rel_path: str) -> Chunk:
        content = src.strip()
        body = f"# {repo} :: {rel_path}\n{content}"
        return Chunk(
            chunk_id=_chunk_id(repo, body),
            doc_id=repo,
            source_uri=f"https://github.com/{repo}/blob/HEAD/{rel_path}",
            modality=Modality.TEXT,
            section_path=[repo, rel_path],
            page=None,
            content=body,
            content_hash=_hash(content),
            metadata={"repo": repo, "file": rel_path, "language": "markdown"},
        )
