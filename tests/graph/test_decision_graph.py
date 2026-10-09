"""The Stage 3 slice end to end, with scripted models.

The live gate is `make decide`; what is tested here is the wiring and the
failure directions -- §5's rule that a degraded run produces a HOLD and never
a trade, and §1's rule that the JSONL is written whatever happens.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.graph.build import build_decision_graph
from research_desk.llm.router import LLMResponse, LLMRouter, ModelProfile
from research_desk.models.market import MarketSnapshot
from research_desk.models.modes import RunMode
from research_desk.models.state import DecisionState

AS_OF = date(2026, 10, 6)

GOOD_REPORT = json.dumps({
    "kind": "market", "stance": "bullish", "confidence": 0.7,
    "summary": "Price is extended above every moving average.",
    "key_points": ["22.6% above the 200-day SMA",
                   "94th percentile of its 52-week range",
                   "insider selling of $64m"],
    "evidence": [], "data_gaps": [],
})
GOOD_PROPOSAL = json.dumps({
    "action": "BUY", "conviction": 0.7, "target_weight_pct": 8.0,
    "horizon_days": 45,
    "rationale": "Trend is intact and relative strength against the sector is positive.",
    "invalidation": "A close below the 200-day simple moving average at 433.",
    "strongest_counterargument":
        "Twelve-month momentum is negative and the move may be exhausted.",
    "intent_alignment": "",
})


class ScriptedRouter:
    """One canned reply per node."""

    def __init__(self, replies: dict[str, str]):
        self._replies = replies
        self.profiles = {"quick": ModelProfile(name="quick", provider="s", model="m")}
        self.node_routes = {"market_analyst": "quick", "trader": "quick"}
        self.clients = {}
        self.calls: list[str] = []

    def profile_for(self, node: str) -> ModelProfile:
        return self.profiles["quick"]

    async def complete(self, node, messages, schema=None):
        self.calls.append(node)
        return LLMResponse(text=self._replies[node], model="m",
                           profile="quick", provider="s")

    async def aclose(self):
        return None


class FakeRegistry:
    def __init__(self, snapshot: MarketSnapshot | None, error: Exception | None = None):
        self._snapshot = snapshot
        self._error = error

    async def daily_bars(self, symbol, as_of):
        raise AssertionError("snapshot is stubbed at a higher level")


def fixture_book():
    """A book marked on this test's own ``AS_OF``, not the shipped one.

    **These tests must not read data/portfolio.yaml, and the reason is a bug
    they caught.** They hardcode ``AS_OF`` in the past; the shipped book is
    re-marked whenever `make portfolio-refresh` runs. The day the book moved to
    2026-10-07 while AS_OF stayed 2026-10-06, compliance correctly blocked every
    order with "the marks are 1 day(s) in the FUTURE -- look-ahead bias, not
    freshness" and three wiring tests went red for a reason that had nothing to
    do with wiring.

    The guard was right. The tests were wrong to depend on a file that moves, so
    they get their own two-position book instead. Anything asserting against the
    *real* book belongs in tests/intent/, where that coupling is the point.
    """
    from research_desk.models.portfolio import PortfolioSnapshot, Position

    return PortfolioSnapshot(
        as_of=AS_OF, cash=60_000.0, source="seed",
        positions=[
            Position(symbol="MSFT", quantity=20, avg_cost=400.0,
                     last_price=500.0, marked_on=AS_OF),
            Position(symbol="NVDA", quantity=50, avg_cost=200.0,
                     last_price=240.0, marked_on=AS_OF),
        ],
    )


async def run_graph(tmp_path, *, replies=None, snapshot=None, registry_error=None):
    """Drive the graph with prefetch stubbed, so no network is touched."""
    import research_desk.graph.nodes.compliance as compliance_module
    import research_desk.graph.nodes.prefetch as prefetch_module
    from research_desk.providers.base import ProviderError

    settings = Settings(data_dir=tmp_path)
    router = ScriptedRouter(replies or {"market_analyst": GOOD_REPORT,
                                        "trader": GOOD_PROPOSAL})
    ctx = NodeContext(settings=settings, mode=RunMode.LIVE, as_of=AS_OF,
                      extras={"router": router, "registry": object()})

    async def fake_prefetch(state, c):
        if registry_error:
            from research_desk.models.state import NodeError
            return {"errors": [NodeError(node="prefetch", kind="no_data",
                                         message=str(registry_error))]}
        return {"snapshot": snapshot or MarketSnapshot(symbol=state.symbol, as_of=AS_OF),
                "notes": ["prefetch: stubbed"]}

    original = prefetch_module.prefetch
    original_book = compliance_module.load_portfolio
    prefetch_module.prefetch = fake_prefetch
    compliance_module.load_portfolio = lambda **kw: fixture_book()
    try:
        graph = build_decision_graph(ctx)
        result = await graph.ainvoke(DecisionState(
            run_id="testrun", symbol="MSFT", as_of=AS_OF, mode=RunMode.LIVE,
        ))
    finally:
        prefetch_module.prefetch = original
        compliance_module.load_portfolio = original_book

    return DecisionState.model_validate(result), settings, router


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


async def test_the_slice_runs_in_order_and_decides(tmp_path) -> None:
    state, settings, router = await run_graph(tmp_path)

    assert router.calls == ["market_analyst", "trader"], "two LLM nodes, in order"
    assert state.analyst_reports["market"].stance == "bullish"
    assert state.trader_proposal is not None
    decision = state.final_decision
    assert decision is not None
    assert decision.action == "BUY"
    assert decision.symbol == "MSFT"
    assert not decision.degraded


async def test_dissent_survives_into_the_decision(tmp_path) -> None:
    """§4 lists dissent as REQUIRED because it is what the human sees at the
    gate. There is no bear researcher yet, so the trader supplies it."""
    state, _, _ = await run_graph(tmp_path)
    assert "momentum is negative" in state.final_decision.dissent.lower()
    assert state.final_decision.invalidation


async def test_both_artefacts_are_written(tmp_path) -> None:
    """§1: the JSONL is the source of truth for replay; proposal.json is the
    file the §0 split communicates through. Two mechanisms, two jobs."""
    state, settings, _ = await run_graph(tmp_path)

    jsonl = settings.journal_dir / f"decisions_{AS_OF:%Y%m%d}.jsonl"
    assert jsonl.exists()
    record = json.loads(jsonl.read_text().splitlines()[-1])
    assert record["symbol"] == "MSFT"
    assert record["llm_calls"], "raw responses must be in the JSONL for replay_llm"

    proposals = list(settings.proposals_dir.glob("MSFT_*.json"))
    assert len(proposals) == 1
    payload = json.loads(proposals[0].read_text())
    assert payload["decision"]["action"] == "BUY"
    assert payload["run_id"] == state.run_id
    # expires_at is what stops Tuesday's proposal executing on Thursday (§9).
    assert payload["decision"]["expires_at"]


async def test_llm_calls_accumulate_across_nodes(tmp_path) -> None:
    state, _, _ = await run_graph(tmp_path)
    assert len(state.llm_calls) == 2
    assert {r.node for r in state.llm_calls} == {"market_analyst", "trader"}


# --------------------------------------------------------------------------- #
# Failure direction is always HOLD (§5)
# --------------------------------------------------------------------------- #


async def test_a_failed_prefetch_still_writes_a_hold(tmp_path) -> None:
    """A run that produced no file is indistinguishable from one that never
    happened, so the artefact is written either way -- carrying a HOLD."""
    state, settings, _ = await run_graph(
        tmp_path, registry_error=RuntimeError("no cached bars for MSFT"))

    decision = state.final_decision
    assert decision is not None
    assert decision.action == "HOLD"
    assert decision.target_weight_pct == 0.0
    assert decision.conviction == 0.0
    assert decision.degraded is True
    assert list(settings.proposals_dir.glob("MSFT_*.json"))


async def test_an_unparseable_trader_degrades_to_hold(tmp_path) -> None:
    state, _, _ = await run_graph(
        tmp_path,
        replies={"market_analyst": GOOD_REPORT, "trader": "I think you should buy."},
    )
    decision = state.final_decision
    assert decision.action == "HOLD"
    assert decision.degraded is True
    # The dissent slot still says something rather than being empty.
    assert decision.dissent


async def test_a_degraded_buy_is_never_carried_to_the_gate(tmp_path) -> None:
    """Belt and braces on §5. If anything errored, a BUY must not survive --
    `from_proposal` copies the action, so persist overrides it.
    """
    state, _, _ = await run_graph(
        tmp_path,
        replies={"market_analyst": "not json at all", "trader": GOOD_PROPOSAL},
    )
    assert state.errors, "the analyst failure must be recorded"
    assert state.final_decision.action == "HOLD", (
        "an upstream failure must not leave a tradeable decision"
    )
    assert "degraded" in state.final_decision.rationale.lower()


async def test_a_degraded_analyst_is_flagged_to_the_trader(tmp_path) -> None:
    """A trader that treats a degraded report as a neutral read is reasoning
    from nothing and does not know it."""
    from research_desk.graph.nodes.trader import _render_report
    from research_desk.models.state import AnalystReport

    state = DecisionState(
        run_id="t", symbol="MSFT", as_of=AS_OF,
        analyst_reports={"market": AnalystReport.degraded("no output", kind="market")},
    )
    rendered = _render_report(state)
    assert "DEGRADED" in rendered
    assert "not as neutral evidence" in rendered
