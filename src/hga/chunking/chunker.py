"""Layout-aware semantic chunking for markdown/text.

Splits Docling's markdown export into heading-tracked sections, then chunks each
section on paragraph boundaries targeting ~500 tokens (approximated as chars).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Section:
    """A heading-delimited section with its hierarchy path and body text."""

    path: list[str]
    text: str


def split_markdown_sections(md: str) -> list[Section]:
    """Split markdown by headings, tracking the hierarchy path of each section."""
    matches = list(_HEADING_RE.finditer(md))
    sections: list[Section] = []

    if not matches:
        body = md.strip()
        return [Section(path=["(document)"], text=body)] if body else []

    if matches[0].start() > 0:
        pre = md[: matches[0].start()].strip()
        if pre:
            sections.append(Section(path=["(preamble)"], text=pre))

    bounds = [m.end() for m in matches] + [len(md)]
    path: list[str] = []
    prev_level = 0
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while path and prev_level >= level:
            path.pop()
            prev_level -= 1
        path.append(title)
        prev_level = level
        sections.append(Section(path=list(path), text=md[bounds[i]: bounds[i + 1]]))
    return sections


def chunk_text(text: str, target_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Greedy paragraph-aware split into chunks of ~target_chars with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > target_chars:
            chunks.append(cur.strip())
            tail = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (tail + "\n\n" + para) if tail else para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
