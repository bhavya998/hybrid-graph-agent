"""Eval harness — LLM-as-judge + retrieval metrics against a golden set.

Loads ``eval/golden/golden.jsonl``, runs each query through the agent, computes
faithfulness / answer relevancy / context precision / retrieval rank metrics,
and writes an ``eval/reports/`` JSON report that doubles as a CI regression gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from hga.agents.graph import AgentRunner
from hga.config import Settings, get_settings
from hga.llm import get_llm_client

_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

_FAITHFULNESS_SYS = (
    "You are an evaluator. Rate whether the ANSWER is faithful to the CONTEXT"
    " (0=completely fabricated, 1=fully supported by context)."
)

_ANSWER_RELEVANCY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

_ANSWER_RELEVANCY_SYS = (
    "You are an evaluator. Rate whether the ANSWER directly addresses the QUERY"
    " (0=completely off-topic, 1=fully relevant)."
)


class EvalHarness:
    """Runs the evaluation pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._agent = AgentRunner(settings=self.s)
        self._judge = get_llm_client(self.s)

    def run(self, golden_path: str | None = None, report_dir: str | None = None) -> dict[str, Any]:
        golden = Path(golden_path or "eval/golden/golden.jsonl")
        if not golden.exists():
            return {"error": f"Golden file not found: {golden}"}
        items = [
            json.loads(line)
            for line in golden.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        import asyncio
        results = asyncio.run(self._run_all(items))

        report_dir_p = Path(report_dir or "eval/reports")
        report_dir_p.mkdir(parents=True, exist_ok=True)
        report = report_dir_p / "latest.json"
        report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results

    async def _run_all(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        per_item: list[dict[str, Any]] = []
        for item in items:
            query = item["query"]
            expected = item.get("expected_chunks", [])
            state = await self._agent.ask(query)
            answer = state.get("draft_answer", "")
            chunks = state.get("retrieved_chunks", [])
            chunk_ids = [c.get("chunk_id", "") for c in chunks]

            faithfulness = await self._judge_faithfulness(query, answer, chunks)
            relevancy = await self._judge_relevancy(query, answer)
            precision, recall = self._context_precision_recall(chunk_ids, expected)
            ndcg, mrr, hit = self._retrieval_metrics(chunk_ids, expected)

            per_item.append({
                "query": query,
                "answer": answer[:500],
                "expected_chunks": expected,
                "retrieved_chunks": chunk_ids,
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_precision": round(precision, 3),
                "context_recall": round(recall, 3),
                "ndcg": round(ndcg, 3),
                "mrr": round(mrr, 3),
                "hit_rate_5": round(hit, 3),
            })

        n = len(per_item)
        return {
            "n_items": n,
            "macro_faithfulness": _mean(per_item, "faithfulness"),
            "macro_relevancy": _mean(per_item, "answer_relevancy"),
            "macro_context_precision": _mean(per_item, "context_precision"),
            "macro_context_recall": _mean(per_item, "context_recall"),
            "macro_ndcg": _mean(per_item, "ndcg"),
            "macro_mrr": _mean(per_item, "mrr"),
            "macro_hit_rate_5": _mean(per_item, "hit_rate_5"),
            "items": per_item,
        }

    async def _judge_faithfulness(
        self, query: str, answer: str, chunks: list[dict[str, Any]]
    ) -> float:
        context = "\n---\n".join(
            c.get("payload", {}).get("content", "")[:600] for c in chunks[:4]
        )
        if not context or not answer:
            return 0.0
        prompt = f"QUERY: {query}\nANSWER: {answer}\nCONTEXT:\n{context}"
        try:
            r = await self._judge.complete_json(
                [{"role": "user", "content": prompt}],
                schema=_FAITHFULNESS_SCHEMA,
                system=_FAITHFULNESS_SYS,
                max_tokens=128,
                temperature=0.0,
            )
            return float(r.get("score", 0.0))
        except Exception:
            return 0.0

    async def _judge_relevancy(self, query: str, answer: str) -> float:
        if not answer:
            return 0.0
        prompt = f"QUERY: {query}\nANSWER: {answer}"
        try:
            r = await self._judge.complete_json(
                [{"role": "user", "content": prompt}],
                schema=_ANSWER_RELEVANCY_SCHEMA,
                system=_ANSWER_RELEVANCY_SYS,
                max_tokens=128,
                temperature=0.0,
            )
            return float(r.get("score", 0.0))
        except Exception:
            return 0.0

    def _context_precision_recall(
        self, retrieved: list[str], expected: list[str]
    ) -> tuple[float, float]:
        ret_set = set(retrieved)
        exp_set = set(expected)
        if not exp_set:
            return (0.0, 0.0)
        if not ret_set:
            return (0.0, 0.0)
        tp = len(ret_set & exp_set)
        precision = tp / len(ret_set)
        recall = tp / len(exp_set)
        return (precision, recall)

    def _retrieval_metrics(
        self, retrieved: list[str], expected: list[str]
    ) -> tuple[float, float, float]:
        if not expected:
            return (0.0, 0.0, 0.0)
        exp_set = set(expected)
        dcg, mrr, hit = 0.0, 0.0, 0.0
        top5 = retrieved[:5]
        found_relevant = False
        for i, cid in enumerate(retrieved):
            if cid in exp_set:
                rank = i + 1
                dcg += 1.0 / math.log2(1 + rank)
                if not found_relevant:
                    mrr = 1.0 / rank
                    found_relevant = True
        for cid in top5:
            if cid in exp_set:
                hit = 1.0
                break
        n_rel = min(len(expected), len(retrieved))
        ideal_dcg = sum(1.0 / math.log2(1 + i) for i in range(1, n_rel + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        return (ndcg, mrr, hit)


def _mean(items: list[dict[str, Any]], key: str) -> float:
    vals = [float(item.get(key, 0.0)) for item in items]
    return sum(vals) / len(vals) if vals else 0.0
