"""Chainlit chat UI — interactive agent conversation with citations.

Run:    uv run chainlit run src/hga/serving/chat.py -w
"""

from __future__ import annotations

import chainlit as cl

from hga.agents.graph import AgentRunner

_agent: AgentRunner | None = None


def _get_runner() -> AgentRunner:
    global _agent  # noqa: PLW0603
    if _agent is None:
        _agent = AgentRunner()
    return _agent


@cl.on_chat_start
async def on_start() -> None:
    await cl.Message(content="Hybrid RAG + GraphRAG ready. Ask about the papers or code.").send()


@cl.on_message
async def on_message(msg: cl.Message) -> None:
    runner = _get_runner()
    state = await runner.ask(msg.content)

    answer = state.get("draft_answer", "No answer produced.")
    citations = state.get("citations", []) or []
    route = state.get("route", "")
    retries = state.get("retry_count", 0)

    meta = f"route={route}  retries={retries}  citations={len(citations)}"
    elements = []
    for i, c in enumerate(citations[:8], 1):
        doc = c.get("doc_id", "?")
        cid = c.get("chunk_id", "?")
        pg = c.get("page", "?")
        elements.append(
            cl.Text(
                name=f"Source {i}",
                content=f"doc: {doc}\nid: {cid}\npage: {pg}",
                display="inline",
            )
        )
    await cl.Message(content=f"{answer}\n\n_{meta}_", elements=elements).send()
