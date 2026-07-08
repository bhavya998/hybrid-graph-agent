"""FastAPI application — exposes the agent via a simple REST API.

POST /ask           — run a query through the agent graph
GET  /health        — liveness check
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hga.agents.graph import AgentRunner

if TYPE_CHECKING:
    from fastapi import FastAPI

runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    """Lazy-init the agent runner (expensive constructor, reuse across requests)."""
    global runner  # noqa: PLW0603
    if runner is None:
        runner = AgentRunner()
    return runner


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    route: str
    retry_count: int


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="hga — Hybrid Graph Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        r = get_runner()
        state = await r.ask(req.query)
        return AskResponse(
            answer=state.get("draft_answer", ""),
            citations=state.get("citations", []) or [],
            route=str(state.get("route", "hybrid")),
            retry_count=state.get("retry_count", 0),
        )

    return app


app = create_app()
