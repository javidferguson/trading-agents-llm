"""The four analysts (§3 nodes 2-5), built from one factory.

They differ only in which slice of the snapshot they read and which prompt
they use, so they are one function parameterised rather than four near-copies
that drift. **All four are intent-blind (§8)** -- the factory never touches
portfolio intent, which is why `tests/test_intent_blindness.py` can check the
whole family by checking this file.

Node 4 is `positioning`, not `social`. §3: it reads short interest,
short-sale volume and insider transactions -- *"all free, all point-in-time,
all far better behaved than retail social sentiment"*. `social_analyst`
survives in the design as a swappable alternative occupant of the same slot,
shipping disabled, and turns on only if an ablation shows lift.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from ...context import NodeContext
from ...llm.structured import structured
from ...metrics.render import has_facts_for, render_for_analyst
from ...models.state import (
    AnalystReport,
    DecisionState,
    ErrorSeverity,
    NodeError,
    NodePatch,
)
from ...prompts import load_prompt

logger = logging.getLogger(__name__)

#: kind -> node name in models.yaml. The four that run as a parallel fan-out.
ANALYSTS: dict[str, str] = {
    "market": "market_analyst",
    "news": "news_analyst",
    "positioning": "positioning_analyst",
    "fundamentals": "fundamentals_analyst",
}


def make_analyst(kind: str) -> Callable[[DecisionState, NodeContext], Awaitable[NodePatch]]:
    """Build one analyst node for ``kind``."""
    node = ANALYSTS[kind]

    async def analyst(state: DecisionState, ctx: NodeContext) -> NodePatch:
        router = ctx.extras.get("router")
        if router is None:
            return {"errors": [NodeError(node=node, kind="misconfigured",
                                         message="no LLM router on the context")]}

        if state.snapshot is None:
            return _degraded(node, kind, "prefetch produced no snapshot")

        # An empty slice is not worth ~30s of local inference. A model handed
        # nothing will still produce a confident report about nothing, which
        # is strictly worse than an honest absence -- and §15.2's whole worry
        # is manufactured agreement.
        if not has_facts_for(state.snapshot, kind):
            reason = f"no {kind} data was available for this symbol"
            logger.info("%s: skipping the model call -- %s", node, reason)
            # PARTIAL, and the only partial emit site in the codebase. An
            # absent input narrows the decision; it does not void it. See
            # NodeError.severity for the 31 runs that proved the difference.
            return _degraded(node, kind, reason, severity="partial")

        budget = ctx.extras.get("budget")
        if budget is not None and (spent := budget.exhausted()):
            return _degraded(node, kind, f"budget exhausted before this node: {spent}")

        result = await structured(
            router, node, AnalystReport,
            [
                {"role": "system", "content": load_prompt(node)},
                # Its own slice and nothing else. See render.ANALYST_BLOCKS.
                {"role": "user", "content": render_for_analyst(state.snapshot, kind)},
            ],
            degraded_fields={"kind": kind},
        )
        if budget is not None:
            budget.record(calls=len(result.records), usd=result.usd)

        patch: NodePatch = {
            "analyst_reports": {kind: result.value},
            "llm_calls": result.records,
            "notes": [
                f"{node}: {result.value.stance} @ {result.value.confidence:.2f}"
                + (" (degraded)" if result.degraded else "")
            ],
        }
        if result.degraded:
            patch["errors"] = [NodeError(
                node=node, kind="degraded",
                message=result.value.degraded_reason or "unknown",
            )]
        return patch

    analyst.__name__ = node
    return analyst


def _degraded(
    node: str, kind: str, reason: str, *, severity: ErrorSeverity = "fatal",
) -> NodePatch:
    """A missing analyst is recorded, never silently absent.

    Downstream has to be able to tell "this analyst had nothing to say" from
    "this analyst was never run" -- the researchers weight reports, and an
    absent one must not read as a neutral one.

    **``severity`` is passed explicitly, and defaults to fatal**, because the
    three callers mean three different things under one ``kind="no_data"``:

    * an empty slice -- ``partial``. The input was absent.
    * no snapshot at all -- fatal. There are no prices; ``prefetch`` has
      already said so in its own error, and this agrees with it.
    * **budget exhausted -- fatal.** §5 names budget exhaustion as a HOLD
      trigger in the same breath as a node error, and it is one: the analyst
      would have had data, and the run ran out of money instead. Nothing about
      the symbol is missing, so treating it as absent evidence would hide a
      machinery limit as a data gap.
    """
    return {
        "analyst_reports": {kind: AnalystReport.degraded(reason, kind=kind)},
        "errors": [NodeError(node=node, kind="no_data", message=reason,
                             severity=severity)],
        "notes": [f"{node}: degraded -- {reason}"],
    }
