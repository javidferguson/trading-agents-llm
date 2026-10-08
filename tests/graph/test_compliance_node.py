"""Node 14 in the graph: ``compliance``. §3's deterministic veto.

The node's own contract, as distinct from the rules it applies (those are
``tests/intent/test_compliance.py``). What matters here is wiring:

* it is in all three graphs, because a ``proposal.json`` written without it is
  indistinguishable from a checked one;
* a veto reaches ``execute`` as a HOLD, not as a BUY with a blocked plan;
* ``persist`` writes what this node decided rather than promoting again;
* and the node never raises, whatever is missing.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.graph.nodes.compliance import compliance
from research_desk.intent.engine import load_portfolio
from research_desk.models.market import MarketSnapshot
from research_desk.models.state import DecisionState, NodeError, RiskVerdict, TraderProposal

#: Derived from the shipped book rather than hardcoded, deliberately.
#:
#: These tests exercise the node against the REAL data/portfolio.yaml -- some
#: of them read its marks directly -- so they must share its decision date. A
#: fixed date went stale the moment the book was re-marked, and compliance then
#: blocked every order with "the marks are 1 day(s) in the FUTURE -- look-ahead
#: bias, not freshness", which is the guard being right and the test being
#: wrong. Deriving it means `make portfolio-refresh` cannot break these again.
#:
#: Wiring tests that do NOT want this coupling use their own fixture book --
#: see `fixture_book()` in tests/graph/test_decision_graph.py.
#: The seeded book's mark date, which is the newest cached bar. Derived rather
#: than hardcoded: compliance treats marks dated after the decision as
#: look-ahead bias and blocks, so a fixed date goes stale the moment `make bars`
#: runs. (It was `load_portfolio().as_of` until that file became the broker's.)
def _seed_as_of():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "seed_portfolio.py"
    spec = importlib.util.spec_from_file_location("seed_portfolio_asof", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_portfolio_asof"] = module
    spec.loader.exec_module(module)
    try:
        return module.seed().as_of
    except SystemExit:
        return date.today()


AS_OF = _seed_as_of()


def ctx(tmp_path) -> NodeContext:
    return NodeContext(settings=Settings(data_dir=tmp_path), as_of=AS_OF,
                       extras={"registry": None})


@pytest.fixture(autouse=True)
def _book(monkeypatch, seeded_book):
    """Give the node the SEEDED book, not ``data/portfolio.yaml``.

    The node calls ``load_portfolio()`` itself, so the fixture has to be
    injected there. Without this the tests assert against the live account:
    once Stage 7's ``execute`` wrote the real (all-cash) book over the seed,
    four of them failed because the book had fifteen free slots and the symbol
    they expected to be blocked by ``max_positions`` no longer was.

    The file is state. The seed recipe is the fixture.
    """
    import research_desk.graph.nodes.compliance as node

    monkeypatch.setattr(node, "load_portfolio", lambda **kw: seeded_book)
    return seeded_book


def proposal(action="BUY", weight=7.0, conviction=0.8) -> TraderProposal:
    return TraderProposal(
        action=action, conviction=conviction, target_weight_pct=weight,
        horizon_days=60,
        rationale="Relative strength is improving against the sector.",
        invalidation="A close below the 200-day moving average would end this.",
        strongest_counterargument="Twelve-month momentum is still negative.",
    )


def verdict(action="BUY", weight=7.0, conviction=0.8, decision="approve") -> RiskVerdict:
    return RiskVerdict(
        decision=decision, action=action, conviction=conviction,
        target_weight_pct=weight, horizon_days=60,
        rationale="The committee is satisfied with the risk on this position.",
        dissent="A liquidity shock would hit this name harder than the index.",
        invalidation="A close below the 200-day moving average would end this.",
    )


def snapshot(symbol="TSM", close: float | None = 484.79,
             adv: float | None = 2e9) -> MarketSnapshot:
    snap = MarketSnapshot(symbol=symbol, as_of=AS_OF)
    trend = snap.trend.model_copy(update={"close": close})
    risk = snap.risk.model_copy(update={"dollar_adv_20d": adv})
    return snap.model_copy(update={"trend": trend, "risk": risk})


def state(**overrides) -> DecisionState:
    body = {
        "run_id": "t", "symbol": "TSM", "as_of": AS_OF,
        "snapshot": snapshot(), "trader_proposal": proposal(),
    }
    body.update(overrides)
    return DecisionState(**body)


# --------------------------------------------------------------------------- #
# What the node returns
# --------------------------------------------------------------------------- #


async def test_the_node_produces_a_plan_a_decision_and_the_book(tmp_path) -> None:
    patch = await compliance(state(), ctx(tmp_path))
    assert patch["order_plan"] is not None
    assert patch["final_decision"] is not None
    assert patch["portfolio"] is not None
    assert patch["drift"] is not None


async def test_the_node_sizes_against_the_snapshots_fresh_price(tmp_path) -> None:
    """The run's own mark is preferred over the book's stored one."""
    patch = await compliance(
        state(snapshot=snapshot(close=500.0)), ctx(tmp_path)
    )
    assert patch["order_plan"].reference_price == pytest.approx(500.0)


async def test_the_node_falls_back_to_the_books_mark(tmp_path, seeded_book) -> None:
    """Which is the reason a ``Position`` stores a price at all: ``decide``
    builds a snapshot for ONE symbol and has no quote for the others."""
    patch = await compliance(
        state(snapshot=snapshot(close=None)), ctx(tmp_path)
    )
    expected = seeded_book.get("TSM").last_price
    assert patch["order_plan"].reference_price == pytest.approx(expected)


async def test_the_fund_managers_verdict_wins_over_the_traders_proposal(tmp_path) -> None:
    patch = await compliance(
        state(trader_proposal=proposal(action="BUY", weight=9.0),
              risk_verdict=verdict(action="HOLD", weight=0.0, decision="veto")),
        ctx(tmp_path),
    )
    assert patch["final_decision"].action == "HOLD"


async def test_the_traders_proposal_is_promoted_when_there_is_no_verdict(tmp_path) -> None:
    """The ``--slice`` and ``--research`` graphs stop before the risk committee."""
    patch = await compliance(state(risk_verdict=None), ctx(tmp_path))
    assert patch["final_decision"].action == "BUY"


# --------------------------------------------------------------------------- #
# A veto reaches `execute` as a HOLD
# --------------------------------------------------------------------------- #


async def test_a_vetoed_order_becomes_a_hold_in_the_final_decision(tmp_path) -> None:
    """Not a BUY with a blocked plan attached.

    PLTR rather than a held name: it is themed and in-universe but UNHELD, and
    the seeded book sits at max_positions, so the block has a real cause. (This
    was GOOGL until the 2026-10-07 widening put GOOGL in the book.)

    Both objects go to the journal, but only one is what the next process acts
    on, and it must not be possible to read a BUY out of a run Python refused.
    """
    # PLTR is themed and in-universe but unheld, and the book is at
    # max_positions -- so this is blocked for a real reason.
    patch = await compliance(
        state(symbol="PLTR", snapshot=snapshot("PLTR", close=182.40),
              trader_proposal=proposal(weight=5.0, conviction=1.0)),
        ctx(tmp_path),
    )
    plan = patch["order_plan"]
    assert plan.blocked
    assert "max_positions" in [v.rule for v in plan.blocking]
    assert patch["final_decision"].action == "HOLD"
    assert patch["final_decision"].target_weight_pct == 0.0
    assert plan.quantity == 0


async def test_a_veto_is_not_recorded_as_a_failure(tmp_path) -> None:
    """**A veto is a judgement, and must not file itself as a malfunction.**

    ``execute`` reads ``FinalDecision.degraded`` to tell *"that a HOLD was a
    failure rather than a judgement"* (§4). The first live Stage 6 run reported
    a correct veto as "DEGRADED -- this HOLD is a failure, not a judgement",
    which is exactly backwards: the veto is the one place in this pipeline
    where the system is working as designed.

    The violations are still fully recorded -- on the ``OrderPlan``, in the
    notes, and in ``proposal.json`` -- so nothing is hidden. What changes is
    that a blocked order no longer looks like a crash.
    """
    patch = await compliance(
        state(symbol="PLTR", snapshot=snapshot("PLTR", close=182.40),
              trader_proposal=proposal(weight=5.0, conviction=1.0)),
        ctx(tmp_path),
    )
    assert patch["order_plan"].blocked
    assert not patch["final_decision"].degraded
    assert "errors" not in patch
    # Recorded, just not as a failure.
    assert patch["order_plan"].blocking
    assert any("BLOCK" in note for note in patch["notes"])


async def test_a_veto_on_an_ALREADY_degraded_run_stays_degraded(tmp_path) -> None:
    """The flag tracks the node failure, not the veto -- so a run that both
    degraded and got vetoed is still reported as degraded."""
    patch = await compliance(
        state(symbol="PLTR", snapshot=snapshot("PLTR", close=182.40),
              trader_proposal=proposal(weight=5.0, conviction=1.0),
              errors=[NodeError(node="news_analyst", kind="degraded",
                                message="no usable output")]),
        ctx(tmp_path),
    )
    assert patch["final_decision"].degraded


async def test_every_violation_appears_in_the_notes(tmp_path) -> None:
    """The notes are what `desk decide` prints and what a human skims."""
    patch = await compliance(
        state(symbol="PLTR", snapshot=snapshot("PLTR", close=182.40),
              trader_proposal=proposal(weight=5.0, conviction=1.0)),
        ctx(tmp_path),
    )
    rendered = "\n".join(patch["notes"])
    assert "BLOCK" in rendered
    assert "max_positions" in rendered


# --------------------------------------------------------------------------- #
# Failure direction is HOLD (§5)
# --------------------------------------------------------------------------- #


async def test_a_degraded_run_cannot_carry_a_trade_into_sizing(tmp_path) -> None:
    patch = await compliance(
        state(errors=[NodeError(node="market_analyst", kind="degraded",
                                message="no usable output")]),
        ctx(tmp_path),
    )
    assert patch["final_decision"].action == "HOLD"
    assert patch["final_decision"].degraded
    assert patch["order_plan"].quantity == 0


async def test_a_missing_proposal_produces_a_hold_not_a_crash(tmp_path) -> None:
    patch = await compliance(state(trader_proposal=None), ctx(tmp_path))
    assert patch["final_decision"] is None or \
        patch.get("final_decision").action == "HOLD"


async def test_an_unreadable_book_produces_a_hold_not_a_crash(tmp_path, monkeypatch) -> None:
    """No book, no sizing. A HOLD with the reason beats a traceback, and beats
    an unsized order reaching `execute` by a wide margin."""
    import research_desk.graph.nodes.compliance as node

    def boom(**kwargs):
        raise FileNotFoundError("data/portfolio.yaml does not exist")

    monkeypatch.setattr(node, "load_portfolio", boom)
    patch = await compliance(state(), ctx(tmp_path))
    assert "errors" in patch
    assert patch["errors"][0].kind == "misconfigured"
    assert "order_plan" not in patch


async def test_the_node_never_reaches_a_model(tmp_path) -> None:
    """§3 marks node 14 as no-LLM. The context carries no router at all here,
    so a model call would be an AttributeError rather than a silent cost."""
    context = NodeContext(settings=Settings(data_dir=tmp_path), as_of=AS_OF,
                          extras={"registry": None})
    assert "router" not in context.extras
    patch = await compliance(state(), context)
    assert patch["order_plan"] is not None


# --------------------------------------------------------------------------- #
# persist writes what this node decided
# --------------------------------------------------------------------------- #


async def test_persist_does_not_re_promote_over_the_veto(tmp_path) -> None:
    """The one thing in this pipeline that must not be overridable by a later node."""
    from research_desk.graph.nodes.persist import persist

    patch = await compliance(
        state(symbol="PLTR", snapshot=snapshot("PLTR", close=182.40),
              trader_proposal=proposal(weight=5.0, conviction=1.0)),
        ctx(tmp_path),
    )
    held = DecisionState(
        run_id="t", symbol="PLTR", as_of=AS_OF,
        trader_proposal=proposal(weight=5.0, conviction=1.0),
        final_decision=patch["final_decision"],
        order_plan=patch["order_plan"],
        portfolio=patch["portfolio"],
    )
    await persist(held, ctx(tmp_path))

    written = json.loads(
        next((tmp_path / "proposals").glob("PLTR_*.json")).read_text()
    )
    assert written["decision"]["action"] == "HOLD", (
        "persist re-derived the decision and discarded the veto"
    )


async def test_proposal_json_carries_the_order_plan_and_the_book(tmp_path) -> None:
    """§9's middle layer. `execute` needs the share count AND the text a human
    approves; neither is derivable from the other."""
    from research_desk.graph.nodes.persist import persist

    patch = await compliance(state(), ctx(tmp_path))
    held = DecisionState(
        run_id="t", symbol="TSM", as_of=AS_OF,
        final_decision=patch["final_decision"], order_plan=patch["order_plan"],
        portfolio=patch["portfolio"],
    )
    await persist(held, ctx(tmp_path))

    written = json.loads(
        next((tmp_path / "proposals").glob("TSM_*.json")).read_text()
    )
    assert "order_plan" in written
    assert written["order_plan"]["symbol"] == "TSM"
    assert "caps_pct" in written["order_plan"]
    assert written["book"]["equity"] > 0
    assert written["book"]["source"] in {"seed", "cache", "ib"}


# --------------------------------------------------------------------------- #
# The node is in every graph
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "builder",
    ["build_decision_graph", "build_research_graph", "build_full_graph"],
)
def test_compliance_is_wired_into_every_graph(builder: str, tmp_path) -> None:
    """A graph that writes a proposal.json without the veto would produce an
    artefact `execute` cannot tell apart from a checked one."""
    import research_desk.graph.build as build

    graph = getattr(build, builder)(ctx(tmp_path))
    assert "compliance" in graph.get_graph().nodes


@pytest.mark.parametrize(
    "builder",
    ["build_decision_graph", "build_research_graph", "build_full_graph"],
)
def test_compliance_runs_immediately_before_persist(builder: str, tmp_path) -> None:
    """Order matters: persist must see a sized, checked decision."""
    import research_desk.graph.build as build

    graph = getattr(build, builder)(ctx(tmp_path))
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("compliance", "persist") in edges
    assert not any(
        source != "compliance" and target == "persist" for source, target in edges
    ), "something reaches persist without passing through the veto"
