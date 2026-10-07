"""Node 0 -- ``build_market_snapshot()``. No LLM, no judgement (§3).

Everything numeric for the run happens here, before any model is called. §2:
*"deterministic prefetch, then constrained summarization."* The paper's "~20
tool calls" are ``get_price_history``, ``get_news``, ``get_financials`` -- there
is no judgment in choosing them, so they are not choices.

**Stage 6 added the book and the drift table to this node**, for the same
reason the snapshot is here: both are pure arithmetic, both are needed before
any model runs, and computing them once means the trader and the compliance
node cannot disagree about what the book held. Note what that does *not* mean
-- the drift table is in ``DecisionState``, and analysts read the snapshot
through ``render_for_analyst`` which cannot see it. Intent still enters at the
trader (§8), and ``tests/test_intent_blindness.py`` holds that line.

A missing or malformed ``config/portfolio.yaml`` is recorded and the run
continues to a HOLD: the compliance node needs the book and will refuse
without it, which is the correct place for that refusal to be visible.
"""

from __future__ import annotations

import logging

from ...config import load_yaml
from ...context import NodeContext
from ...intent.engine import compute_gaps, load_intent, load_portfolio
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

    patch: NodePatch = {
        "snapshot": snapshot,
        "notes": [f"prefetch: {populated} metrics, {missing} unavailable"],
    }

    try:
        intent = load_intent()
        portfolio = load_portfolio()
    except Exception as exc:  # noqa: BLE001 -- the book is not worth a crash
        patch["errors"] = [NodeError(
            node="prefetch", kind="no_book",
            message=f"could not load the book: {type(exc).__name__}: {exc}",
        )]
        patch["notes"].append(
            "prefetch: no portfolio -- sizing and the drift table are unavailable"
        )
        return patch

    drift = compute_gaps(intent, portfolio, as_of=state.as_of)
    patch["portfolio"] = portfolio
    patch["drift"] = drift
    patch["notes"].append(
        f"prefetch: book marked {portfolio.as_of}, equity "
        f"{portfolio.equity:,.0f} USD, {portfolio.position_count} position(s), "
        f"{drift.slots_free} free slot(s)"
    )
    return patch
