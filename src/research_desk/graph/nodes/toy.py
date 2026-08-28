"""Two throwaway nodes that exercise the Stage 0 exit gate.

The gate is: *a two-node LangGraph runs and both nodes appear as spans in
Langfuse.* Nothing here survives Stage 3 -- these get replaced by ``prefetch``
and ``market_analyst``.

They are still written in the real shape, because the shape is the thing being
proved: a plain ``async def`` taking ``(state, ctx)``, returning a **partial
patch**, importing neither ``langgraph`` nor ``langfuse``, and doing all of its
arithmetic in Python.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ...context import NodeContext
from ...models.state import DecisionState, LLMCallRecord, NodePatch


async def toy_prefetch(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Stand-in for ``build_market_snapshot()``. No LLM, no network."""
    return {
        "notes": [f"prefetch: {state.symbol} as_of={state.as_of.isoformat()}"],
    }


async def toy_summarise(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Stand-in for an analyst node.

    It records an ``LLMCallRecord`` without calling a model, which is the point:
    it proves the record lands on the state and survives the reducer before
    there is a router to produce a real one.
    """
    prompt = f"summarise {state.symbol} @ {state.as_of.isoformat()}"
    raw = f'{{"stance": "neutral", "symbol": "{state.symbol}"}}'

    record = LLMCallRecord(
        node="toy_summarise",
        profile="none",
        provider="stub",
        model="stub",
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        raw_response=raw,
        latency_ms=0,
        usd=0.0,
    )
    return {
        "llm_calls": [record],
        "notes": [f"summarise: {len(state.notes)} note(s) seen upstream"],
    }


def fanout_probe(label: str) -> Any:
    """Build one branch of the reducer test's fan-out.

    Each branch appends exactly one note. If the reducer annotation on
    ``DecisionState.notes`` is missing or wrong, the branches overwrite each
    other and only one note survives -- which is the bug this exists to catch.
    """

    async def probe(state: DecisionState, ctx: NodeContext) -> NodePatch:
        return {"notes": [label]}

    probe.__name__ = f"probe_{label}"
    return probe
