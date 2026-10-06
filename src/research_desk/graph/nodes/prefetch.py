"""Node 0 -- ``build_market_snapshot()``. No LLM, no judgement (§3).

Everything numeric for the run happens here, before any model is called. §2:
*"deterministic prefetch, then constrained summarization."* The paper's "~20
tool calls" are ``get_price_history``, ``get_news``, ``get_financials`` -- there
is no judgment in choosing them, so they are not choices.
"""

from __future__ import annotations

import logging

from ...config import load_yaml
from ...context import NodeContext
from ...metrics.snapshot import build_market_snapshot, universe_context
from ...models.state import DecisionState, NodeError, NodePatch
from ...providers.base import ProviderError

logger = logging.getLogger(__name__)


async def prefetch(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Fetch and compute every metric for ``state.symbol`` as of ``state.as_of``."""
    registry = ctx.extras.get("registry")
    if registry is None:
        return {
            "errors": [NodeError(
                node="prefetch", kind="misconfigured",
                message="no provider registry on the context",
            )]
        }

    benchmark, sector = universe_context(load_yaml("universe.yaml"), state.symbol)

    try:
        snapshot = await build_market_snapshot(
            registry, state.symbol, state.as_of,
            benchmark_symbol=benchmark, sector_symbol=sector,
        )
    except ProviderError as exc:
        # No prices means no report worth writing. Recorded rather than raised
        # so the run degrades to HOLD (§5) instead of dying.
        return {
            "errors": [NodeError(
                node="prefetch", kind="no_data", message=str(exc).splitlines()[0],
            )],
            "notes": [f"prefetch failed for {state.symbol}"],
        }

    populated, missing = snapshot.metric_count()
    logger.info(
        "%s: %d metrics populated, %d unavailable (%d bars)",
        state.symbol, populated, missing, snapshot.bars_available,
    )
    return {
        "snapshot": snapshot,
        "notes": [f"prefetch: {populated} metrics, {missing} unavailable"],
    }
