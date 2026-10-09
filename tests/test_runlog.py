"""``runlog`` -- reading the journal back. §1's source of truth, from the other end.

> *"This JSONL is the source of truth for replay and audit... Do not let a
> future refactor 'simplify' by deleting the JSONL."* (architecture §1)

A source of truth nothing can read is a claim rather than a property, which is
what it was for nine subcommands. These tests cover the two things that would
make the reader quietly useless: a filter that silently returns the wrong runs,
and a damaged file that takes a whole day's history down with it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.models.orders import OrderPlan, Violation
from research_desk.models.portfolio import PortfolioSnapshot, Position
from research_desk.models.state import (
    DecisionState,
    FinalDecision,
    LLMCallRecord,
    NodeError,
)
from research_desk.runlog import (
    journal_files,
    read_runs,
    render_run,
    render_summary,
)

AS_OF = date(2026, 10, 6)


def decision(action="BUY", degraded=False, symbol="TSM") -> FinalDecision:
    return FinalDecision(
        action=action, symbol=symbol, conviction=0.7,
        target_weight_pct=0.0 if action == "HOLD" else 5.0,
        horizon_days=45,
        rationale="relative strength is improving against the sector",
        dissent="a liquidity shock would hit this name harder than the index",
        invalidation="a close below the 200-day moving average at 389",
        expires_at=datetime(2026, 10, 7, 12, 0), degraded=degraded,
    )


def run(symbol="TSM", run_id="20261006-120000-aaaaaa", action="BUY",
        **overrides) -> DecisionState:
    body = {
        "run_id": run_id, "symbol": symbol, "as_of": AS_OF,
        "final_decision": decision(action, symbol=symbol),
        "notes": ["prefetch: 67 metrics", "trader: BUY 5.0%"],
        "llm_calls": [
            LLMCallRecord(node="trader", profile="quick", provider="ollama",
                          model="qwen3:8b", prompt_hash="h", raw_response="{}",
                          latency_ms=1500, usd=0.0),
        ],
    }
    body.update(overrides)
    return DecisionState(**body)


def write(journal_dir: Path, *states: DecisionState, on: date = AS_OF) -> Path:
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"decisions_{on:%Y%m%d}.jsonl"
    with path.open("a") as handle:
        for state in states:
            handle.write(json.dumps(state.model_dump(mode="json"), default=str) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_runs_come_back_newest_first(tmp_path) -> None:
    write(tmp_path,
          run(run_id="20261006-100000-aaa"),
          run(run_id="20261006-110000-bbb"),
          run(run_id="20261006-120000-ccc"))
    got = [r.run_id for r in read_runs(tmp_path)]
    assert got == ["20261006-120000-ccc", "20261006-110000-bbb",
                   "20261006-100000-aaa"]


def test_symbol_filter_is_case_insensitive(tmp_path) -> None:
    write(tmp_path, run(symbol="TSM"), run(symbol="NVDA", run_id="x-2"))
    assert [r.symbol for r in read_runs(tmp_path, symbol="tsm")] == ["TSM"]


def test_run_id_matches_by_prefix(tmp_path) -> None:
    """A timestamp fragment should be enough; nobody copies a full run id."""
    write(tmp_path, run(run_id="20261006-153546-a4ac1a"),
          run(run_id="20261006-200907-18d340"))
    got = read_runs(tmp_path, run_id="20261006-1535")
    assert len(got) == 1
    assert got[0].run_id == "20261006-153546-a4ac1a"


def test_last_caps_the_result(tmp_path) -> None:
    write(tmp_path, *[run(run_id=f"20261006-10000{i}-x") for i in range(5)])
    assert len(read_runs(tmp_path, last=2)) == 2


def test_date_filter_reads_only_that_day(tmp_path) -> None:
    write(tmp_path, run(run_id="day6"), on=date(2026, 10, 6))
    write(tmp_path, run(run_id="day7"), on=date(2026, 10, 7))
    got = read_runs(tmp_path, on=date(2026, 10, 7))
    assert [r.run_id for r in got] == ["day7"]


def test_actionable_keeps_only_buys_and_sells(tmp_path) -> None:
    """Same definition the cadence check uses, so "what did the desk actually
    do today" has one answer rather than two."""
    write(tmp_path,
          run(run_id="a", action="BUY"),
          run(run_id="b", action="HOLD"),
          run(run_id="c", action="SELL"))
    got = {r.run_id for r in read_runs(tmp_path, actionable=True)}
    assert got == {"a", "c"}


def test_days_are_ordered_newest_first_across_files(tmp_path) -> None:
    write(tmp_path, run(run_id="older"), on=date(2026, 10, 5))
    write(tmp_path, run(run_id="newer"), on=date(2026, 10, 7))
    assert [r.run_id for r in read_runs(tmp_path)] == ["newer", "older"]


# --------------------------------------------------------------------------- #
# Damage tolerance
# --------------------------------------------------------------------------- #


def test_a_truncated_final_line_does_not_lose_the_day(tmp_path) -> None:
    """What a run killed mid-write leaves behind. It is normal.

    Discarding the file would mean one interrupted run erases every earlier run
    that day -- from the file the architecture calls the source of truth.
    """
    path = write(tmp_path, run(run_id="intact-1"), run(run_id="intact-2"))
    with path.open("a") as handle:
        handle.write('{"run_id": "trunca')

    got = [r.run_id for r in read_runs(tmp_path)]
    assert got == ["intact-2", "intact-1"]


def test_blank_lines_are_skipped(tmp_path) -> None:
    path = write(tmp_path, run(run_id="one"))
    with path.open("a") as handle:
        handle.write("\n\n")
    assert len(read_runs(tmp_path)) == 1


def test_a_missing_journal_directory_is_empty_not_an_error(tmp_path) -> None:
    assert read_runs(tmp_path / "nope") == []
    assert journal_files(tmp_path / "nope") == []


def test_a_missing_day_is_empty_not_an_error(tmp_path) -> None:
    write(tmp_path, run())
    assert read_runs(tmp_path, on=date(2025, 1, 1)) == []


def test_a_row_the_schema_outgrew_is_skipped_with_a_warning(tmp_path, caplog) -> None:
    """An append-only log outlives the code that wrote it.

    Skipping is the honest behaviour, and the warning is the signal that a
    schema change broke replay -- which §1 cares about more than this reader.
    """
    path = write(tmp_path, run(run_id="good"))
    with path.open("a") as handle:
        handle.write(json.dumps({"run_id": "bad", "symbol": "X"}) + "\n")

    with caplog.at_level("WARNING"):
        got = read_runs(tmp_path)
    assert [r.run_id for r in got] == ["good"]
    assert any("does not validate" in m for m in caplog.messages)


# --------------------------------------------------------------------------- #
# Rendering: the Stage 6 distinction must survive into the reader
# --------------------------------------------------------------------------- #


def test_a_degraded_run_shows_its_reason() -> None:
    state = run(
        action="HOLD",
        final_decision=decision("HOLD", degraded=True),
        errors=[NodeError(node="news_analyst", kind="degraded",
                          message="no usable output")],
    )
    text = render_run(state)
    assert "DEGRADED -- this HOLD is a failure" in text
    assert "news_analyst" in text


def test_a_vetoed_run_shows_the_violations_and_is_not_degraded() -> None:
    """**The distinction fixed in Stage 6, asserted in the reader.**

    A veto is the system working -- "this is where trust comes from" -- so it
    must not read as a crash three weeks later either. The violations are fully
    present; the degraded banner is not.
    """
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="BUY", quantity=0,
        reference_price=484.79,
        violations=[Violation(rule="max_positions", message="the book is full",
                              limit=15, actual=16)],
        skip_reason="blocked by max_positions",
    )
    state = run(action="HOLD",
                final_decision=decision("HOLD", degraded=False),
                order_plan=plan)
    text = render_run(state)

    assert "VETOED BY PYTHON" in text
    assert "max_positions" in text
    assert "DEGRADED" not in text


def test_the_summary_line_distinguishes_vetoed_from_degraded() -> None:
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="HOLD", quantity=0,
        violations=[Violation(rule="portfolio_stale", message="too old")],
    )
    vetoed = render_summary(run(action="HOLD", order_plan=plan))
    degraded = render_summary(
        run(action="HOLD", final_decision=decision("HOLD", degraded=True))
    )
    assert "VETOED:portfolio_stale" in vetoed
    assert "DEGRADED" not in vetoed
    assert "DEGRADED" in degraded


def test_partial_coverage_is_flagged_without_claiming_a_failure() -> None:
    """A sweep exists to show where coverage is thin, and after the severity
    split these runs no longer carry DEGRADED to say so."""
    state = run(final_decision=decision("BUY").model_copy(
        update={"absent_analysts": ["news"]}))
    line = render_summary(state)
    assert "partial:news" in line
    assert "DEGRADED" not in line

    text = render_run(state)
    assert "PARTIAL COVERAGE" in text
    assert "narrower, not untrustworthy" in text


def test_a_degraded_run_outranks_a_coverage_note_in_the_summary() -> None:
    line = render_summary(run(action="HOLD", final_decision=decision(
        "HOLD", degraded=True).model_copy(update={"absent_analysts": ["news"]})))
    assert "DEGRADED" in line
    assert "partial:" not in line


def test_a_warning_is_shown_without_claiming_a_block() -> None:
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="BUY", quantity=11,
        reference_price=472.15, estimated_notional=5193.0,
        violations=[Violation(rule="earnings_blackout", severity="warn",
                              message="could not be evaluated")],
    )
    line = render_summary(run(order_plan=plan))
    assert "warn:earnings_blackout" in line
    assert "VETOED" not in line


def test_the_book_is_rendered_as_it_was_not_as_it_is() -> None:
    """A replay must show the weights the decision was actually made against.

    data/portfolio.yaml is overwritten on every re-mark, so the book is
    recorded in the run rather than referenced.
    """
    book = PortfolioSnapshot(
        as_of=date(2026, 9, 1), cash=10_000.0,
        positions=[Position(symbol="TSM", quantity=10, avg_cost=400.0,
                            last_price=500.0, marked_on=date(2026, 9, 1))],
    )
    text = render_run(run(portfolio=book))
    assert "marked 2026-09-01" in text
    assert "15,000" in text  # 10,000 cash + 5,000 position


def test_full_adds_the_model_calls_and_summary_does_not() -> None:
    state = run()
    assert "MODEL CALLS" not in render_run(state)
    assert "MODEL CALLS" in render_run(state, full=True)


def test_the_cost_line_omits_a_budget_cap_it_cannot_know() -> None:
    """The caps live in models.yaml, not the journal. Printing today's cap next
    to a month-old run would assert something that was never checked."""
    text = render_run(run())
    assert "model call(s)" in text
    assert "budget:" not in text


# --------------------------------------------------------------------------- #
# One renderer, two commands
# --------------------------------------------------------------------------- #


def test_decide_renders_through_runlog_rather_than_its_own_copy() -> None:
    """The reason `runlog` is a module at all.

    The printing used to live inside `cli._run_decide`, so `review` could only
    have copied it and the two would have drifted on the first edit. If this
    import disappears, that has happened.
    """
    source = (Path(__file__).resolve().parents[1]
              / "src" / "research_desk" / "cli.py").read_text()
    assert "from .runlog import render_run" in source
    assert "render_run(state)" in source


def test_rendering_is_deterministic() -> None:
    state = run()
    assert render_run(state) == render_run(state)


# --------------------------------------------------------------------------- #
# Round trip through the real writer
# --------------------------------------------------------------------------- #


async def test_a_run_written_by_persist_reads_back_intact(tmp_path) -> None:
    """The guard against a schema change breaking replay silently.

    `persist` writes; `read_runs` reads. If those two ever disagree, every
    recorded decision becomes unreadable and nothing would say so until Stage 8
    tried to score them.
    """
    from research_desk.graph.nodes.persist import persist

    book = PortfolioSnapshot(
        as_of=AS_OF, cash=50_000.0,
        positions=[Position(symbol="TSM", quantity=10, avg_cost=400.0,
                            last_price=472.15, marked_on=AS_OF)],
    )
    plan = OrderPlan(symbol="TSM", as_of=AS_OF, action="BUY", quantity=11,
                     reference_price=472.15, estimated_notional=5193.65,
                     binding_cap="requested",
                     caps_pct={"requested": 4.5, "cap_intent": 5.0})
    state = run(portfolio=book, order_plan=plan)

    ctx = NodeContext(settings=Settings(data_dir=tmp_path), as_of=AS_OF)
    await persist(state, ctx)

    got = read_runs(tmp_path / "journal")
    assert len(got) == 1
    back = got[0]
    assert back.run_id == state.run_id
    assert back.order_plan.quantity == 11
    assert back.order_plan.binding_cap == "requested"
    assert back.order_plan.caps_pct["requested"] == pytest.approx(4.5)
    # The book must come back as a real object, not a dict -- `desk review`
    # tripped over exactly that, which is why the field is properly typed.
    assert back.portfolio.equity == pytest.approx(54_721.5)
    assert back.portfolio.position_count == 1


def test_the_real_journal_is_readable() -> None:
    """Against the actual accumulated history, not a fixture.

    Cheap, and it is the test that would notice a schema change the fixtures
    above were updated alongside.
    """
    runs = read_runs(Path("data/journal"), last=5)
    if not runs:
        pytest.skip("no journal yet")
    for state in runs:
        assert state.run_id and state.symbol
        assert render_summary(state)
        assert render_run(state)


# --------------------------------------------------------------------------- #
# Intent vs outcome: the sweep's whole reason for existing
# --------------------------------------------------------------------------- #


def verdict(action="BUY", **overrides):
    from research_desk.models.state import RiskVerdict

    body = {
        "decision": "approve", "action": action, "conviction": 0.7,
        "target_weight_pct": 0.0 if action == "HOLD" else 5.0,
        "horizon_days": 45,
        "rationale": "the committee is satisfied with the risk here",
        "dissent": "a liquidity shock would hit this harder than the index",
        "invalidation": "a close below the 200-day moving average at 389",
    }
    body.update(overrides)
    return RiskVerdict(**body)


def test_a_cadence_vetoed_buy_does_not_read_as_a_hold() -> None:
    """**The reason a sweep needed this.**

    `max_decisions_per_day` rewrites everything past the third actionable
    symbol to HOLD, so a 31-symbol research sweep printed ~28 rows saying
    "HOLD" when the fund manager had said BUY. That is the opposite of the
    signal a sweep exists to produce, and the intent was in the journal the
    whole time -- only the renderer dropped it.
    """
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="HOLD", quantity=0,
        violations=[Violation(rule="max_decisions_per_day",
                              message="3 other symbols already decided")],
    )
    state = run(
        action="HOLD",
        final_decision=decision("HOLD"),
        risk_verdict=verdict("BUY"),
        order_plan=plan,
    )

    line = render_summary(state)
    assert "BUY->HOLD" in line, f"intent was dropped: {line}"
    assert "VETOED:max_decisions_per_day" in line


def test_an_unchanged_decision_shows_one_action_not_an_arrow() -> None:
    """The arrow has to mean something. Showing `BUY->BUY` on every row would
    make the rows where Python actually intervened invisible again."""
    state = run(action="BUY", risk_verdict=verdict("BUY"))
    line = render_summary(state)
    assert "->" not in line
    assert "BUY" in line


def test_the_traders_proposal_is_the_intent_when_there_is_no_verdict() -> None:
    """The --slice and --research graphs stop before the risk committee, so the
    trader is the last word there."""
    from research_desk.models.state import TraderProposal
    from research_desk.runlog import wanted_action

    proposal = TraderProposal(
        action="SELL", conviction=0.6, target_weight_pct=2.0, horizon_days=30,
        rationale="relative strength has deteriorated against the sector",
        invalidation="a reclaim of the 50-day moving average",
        strongest_counterargument="the fundamental trend is still intact",
    )
    state = run(action="HOLD", final_decision=decision("HOLD"),
                trader_proposal=proposal)
    assert wanted_action(state) == "SELL"
    assert "SELL->HOLD" in render_summary(state)


def test_a_run_with_no_proposal_at_all_has_no_intent() -> None:
    from research_desk.runlog import wanted_action

    state = run(action="HOLD", final_decision=decision("HOLD"),
                trader_proposal=None)
    assert wanted_action(state) is None
    assert "->" not in render_summary(state)


# --------------------------------------------------------------------------- #
# The --wanted filter
# --------------------------------------------------------------------------- #


def test_wanted_finds_what_actionable_hides(tmp_path) -> None:
    """These two filters must not be the same, and a sweep is where it shows.

    `--actionable` reads ``final_decision`` -- so it hides exactly the rows a
    research sweep is for, every symbol the cadence limit rewrote.
    """
    vetoed = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="HOLD", quantity=0,
        violations=[Violation(rule="max_decisions_per_day", message="full")],
    )
    write(tmp_path,
          run(run_id="placed", action="BUY", risk_verdict=verdict("BUY")),
          run(run_id="vetoed", action="HOLD",
              final_decision=decision("HOLD"),
              risk_verdict=verdict("BUY"), order_plan=vetoed),
          run(run_id="genuine-hold", action="HOLD",
              final_decision=decision("HOLD"), risk_verdict=verdict("HOLD")))

    actionable = {r.run_id for r in read_runs(tmp_path, actionable=True)}
    wanted = {r.run_id for r in read_runs(tmp_path, wanted=True)}

    assert actionable == {"placed"}
    assert wanted == {"placed", "vetoed"}, (
        "the vetoed BUY is what a sweep is looking for and --wanted must find it"
    )
    # And neither should surface a decision nobody wanted to act on.
    assert "genuine-hold" not in wanted


def test_wanted_falls_back_to_the_trader_when_no_verdict_exists(tmp_path) -> None:
    from research_desk.models.state import TraderProposal

    proposal = TraderProposal(
        action="BUY", conviction=0.6, target_weight_pct=3.0, horizon_days=30,
        rationale="relative strength is improving against the sector",
        invalidation="a close below the 200-day moving average",
        strongest_counterargument="momentum could reverse from here",
    )
    write(tmp_path, run(run_id="slice-run", action="HOLD",
                        final_decision=decision("HOLD"),
                        trader_proposal=proposal))
    assert {r.run_id for r in read_runs(tmp_path, wanted=True)} == {"slice-run"}
