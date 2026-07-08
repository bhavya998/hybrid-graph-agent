"""MCP tool client — connects to the hga corpus MCP server (in-process).

The corpus server (hga.tools.corpus_server) is a real FastMCP server that can
be launched standalone via ``mcp run`` or ``uv run python -m hga.tools.corpus_server``.
Any MCP client (Claude Desktop, OpenCode, Cursor) can connect to it.

For the agent, we call the tool functions directly in-process for speed. The
same functions are exported by the server module, so there's no duplication.
"""

from __future__ import annotations

from typing import Any

from hga.tools.corpus_server import (
    list_papers,
    read_paper,
    read_repo_file,
    search_chunks,
    search_corpus,
)

_TOOLS = {
    "search_corpus": search_corpus,
    "read_paper": read_paper,
    "list_papers": list_papers,
    "search_chunks": search_chunks,
    "read_repo_file": read_repo_file,
}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_corpus": {
        "name": "search_corpus",
        "description": (
            "Search paper titles, abstracts, and authors in the downloaded arXiv corpus."
            " Returns matching papers with metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search term to match against titles,"
                        " abstracts, and author names"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "read_paper": {
        "name": "read_paper",
        "description": (
            "Return full metadata for a paper by its arxiv ID: title, all authors,"
            " abstract, categories, github repos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv paper ID, e.g. '2305.06983'"},
            },
            "required": ["arxiv_id"],
        },
    },
    "list_papers": {
        "name": "list_papers",
        "description": "List all papers currently in the corpus with their arxiv IDs and titles.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "search_chunks": {
        "name": "search_chunks",
        "description": (
            "Keyword search across all parsed chunks (text from papers and code)."
            " Returns matching chunk content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search in chunk content",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    "read_repo_file": {
        "name": "read_repo_file",
        "description": (
            "Read the contents of a file from a cloned GitHub repo."
            " Use list_papers first to find repo slugs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_slug": {
                    "type": "string",
                    "description": "GitHub repo slug, e.g. jzbjyb/FLARE",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path relative to the repo root, e.g. src/model.py",
                },
            },
            "required": ["repo_slug", "file_path"],
        },
    },
}


class MCPToolManager:
    """Wraps the hga corpus MCP server tools for in-process agent use.

    The corpus server is a real FastMCP server — launch it standalone with:
        uv run python -m hga.tools.corpus_server
    """

    def __init__(self) -> None:
        self._started = True

    async def start(self, connections: dict[str, Any] | None = None) -> None:
        del connections

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a tool by name. Returns the tool's string result."""
        tool_fn = _TOOLS.get(name)
        if tool_fn is None:
            return f"Tool '{name}' not found. Available: {sorted(_TOOLS)}"
        try:
            args = arguments or {}
            return str(tool_fn(**args))
        except Exception as exc:
            return f"Tool '{name}' error: {exc}"

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata for all available tools (used by agent for tool selection)."""
        return list(_TOOL_SCHEMAS.values())

    async def close(self) -> None:
        pass

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI/Anthropic-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {k: v for k, v in schema.items() if k != "input_schema"}
                | {"parameters": schema["input_schema"]},
            }
            for schema in _TOOL_SCHEMAS.values()
        ]
