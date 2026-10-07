"""The Python veto, rule by rule. §9's second key.

> *"Any violation blocks the order regardless of what the fund manager decided.
> Two-key system: LLM proposes, Python vetoes. This is where trust comes from."*

Every rule §9 names gets a test in which it fires, and one in which it does
not. A veto that has never been seen to fire is indistinguishable from one that
cannot.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from research_desk.intent import compliance as rules
from research_desk.intent.earnings import EarningsEstimate
from research_desk.intent.engine import compute_gaps
from research_desk.intent.sizing import size_order
from research_desk.models.intent import PortfolioIntent
from research_desk.models.orders import OrderPlan, Violation
from research_desk.models.portfolio import PortfolioSnapshot, Position
from research_desk.models.state import FinalDecision

AS_OF = date(2026, 10, 6)
PRICE = 100.0
SECTORS = {"AAA": "XTK", "BBB": "XTK", "CCC": "XFN"}


def intent(**risk_overrides) -> PortfolioIntent:
    risk = {
        "per_trade_risk_pct": 0.75, "default_stop_pct": 8.0,
        "max_position_pct": 10.0, "max_sector_pct": 65.0,
        "min_cash_pct": 10.0, "max_gross_exposure_pct": 95.0,
        "max_positions": 8, "min_trade_usd": 500, "max_order_shares": 2000,
        "max_adv_participation_pct": 2.0,
        "earnings_blackout_days_before": 2, "earnings_blackout_days_after": 1,
    }
    risk.update(risk_overrides)
    return PortfolioIntent.from_dict({
        "objective": {"horizon_days": 60},
        "risk": risk,
        "universe": {"include": ["AAA", "BBB", "CCC"], "exclude": []},
        "themes": [{"name": "T1", "target_weight_pct": 30.0, "conviction": 1.0,
                    "exemplars": ["AAA", "BBB", "CCC"]}],
    })


def book(cash: float, **holdings: tuple[float, float]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=AS_OF, cash=cash,
        positions=[
            Position(symbol=s, quantity=q, avg_cost=p, last_price=p, marked_on=AS_OF)
            for s, (q, p) in holdings.items()
        ],
    )


def decision(action="BUY", weight=10.0, conviction=1.0, symbol="AAA") -> FinalDecision:
    return FinalDecision(
        action=action, symbol=symbol, conviction=conviction,
        target_weight_pct=weight, horizon_days=60,
        rationale="a full sentence of reasoning goes here",
        dissent="the surviving argument against it goes here",
        invalidation="the specific condition that would prove it wrong",
        stop_loss_pct=8.0, expires_at=datetime(2026, 10, 7, 12, 0),
    )


def plan_for(shares: int, symbol="AAA", action="BUY", price=PRICE) -> OrderPlan:
    """A hand-built plan, so a rule can be tested without sizing agreeing."""
    return OrderPlan(
        symbol=symbol, as_of=AS_OF, action=action, quantity=shares,
        reference_price=price, estimated_notional=abs(shares) * price,
    )


def check(plan: OrderPlan, i=None, pf=None, **kwargs) -> OrderPlan:
    kwargs.setdefault("sectors", SECTORS)
    kwargs.setdefault("dollar_adv", 1e12)  # effectively unlimited unless tested
    return rules.check(
        plan, decision(symbol=plan.symbol, action=plan.action),
        i or intent(), pf if pf is not None else book(100_000.0),
        as_of=AS_OF, **kwargs,
    )


# --------------------------------------------------------------------------- #
# post_trade: checks read the state AFTER the order, which is the whole point
# --------------------------------------------------------------------------- #


def test_post_trade_moves_cash_and_the_position_together() -> None:
    pf = book(100_000.0)
    after = rules.post_trade(pf, "AAA", 100, PRICE)
    assert after.position_value("AAA") == pytest.approx(10_000.0)
    assert after.cash == pytest.approx(90_000.0)
    assert after.equity == pytest.approx(pf.equity)


def test_post_trade_averages_the_cost_basis_on_an_add() -> None:
    pf = book(50_000.0, AAA=(100, 100.0))
    after = rules.post_trade(pf, "AAA", 100, 200.0)
    assert after.get("AAA").quantity == 200
    assert after.get("AAA").avg_cost == pytest.approx(150.0)


def test_post_trade_leaves_the_basis_alone_on_a_partial_sale() -> None:
    pf = book(0.0, AAA=(100, 100.0))
    pf = pf.model_copy(update={"positions": [
        Position(symbol="AAA", quantity=100, avg_cost=80.0, last_price=100.0,
                 marked_on=AS_OF)
    ]})
    after = rules.post_trade(pf, "AAA", -50, 100.0)
    assert after.get("AAA").avg_cost == pytest.approx(80.0)


def test_post_trade_removes_a_fully_closed_position() -> None:
    pf = book(0.0, AAA=(100, PRICE))
    after = rules.post_trade(pf, "AAA", -100, PRICE)
    assert after.get("AAA") is None
    assert after.position_count == 0


def test_a_check_on_the_current_state_would_pass_everything() -> None:
    """Why ``post_trade`` exists at all.

    The book is at 9% with a 10% cap, so the CURRENT state is compliant and the
    POST-TRADE state is not. A layer that checked the former would pass every
    order ever proposed.
    """
    pf = book(91_000.0, AAA=(90, PRICE))
    assert pf.weight_pct("AAA") == pytest.approx(9.0)
    result = check(plan_for(50), pf=pf)
    assert result.blocked
    assert [v.rule for v in result.blocking] == ["max_position_pct"]


# --------------------------------------------------------------------------- #
# Each rule §9 names, firing
# --------------------------------------------------------------------------- #


def test_max_position_pct_blocks() -> None:
    result = check(plan_for(200), pf=book(100_000.0))  # 20% of equity
    assert "max_position_pct" in [v.rule for v in result.blocking]


def test_max_sector_pct_blocks() -> None:
    i = intent(max_sector_pct=15.0)
    pf = book(90_000.0, BBB=(100, PRICE))  # BBB is already 10% of XTK
    result = check(plan_for(100), i=i, pf=pf)
    violation = next(v for v in result.blocking if v.rule == "max_sector_pct")
    assert violation.actual == pytest.approx(20.0)
    assert "XTK" in violation.message


def test_min_cash_pct_blocks() -> None:
    i = intent(min_cash_pct=50.0, max_position_pct=100.0)
    result = check(plan_for(600), i=i, pf=book(100_000.0))
    assert "min_cash_pct" in [v.rule for v in result.blocking]


def test_max_gross_exposure_blocks_and_names_the_binding_limit() -> None:
    """In a long-only book the cash floor usually binds before the declared
    gross limit, and the message has to say which one it was."""
    i = intent(max_position_pct=100.0, min_cash_pct=10.0)
    pf = book(100_000.0)
    result = check(plan_for(950), i=i, pf=pf)  # 95% gross, cap is 90%
    violation = next(v for v in result.blocking
                     if v.rule == "max_gross_exposure_pct")
    assert violation.limit == pytest.approx(90.0)
    assert "cash floor" in violation.message


def test_max_positions_blocks_a_new_symbol_on_a_full_book() -> None:
    i = intent(max_positions=1)
    pf = book(90_000.0, BBB=(100, PRICE))
    result = check(plan_for(10, symbol="AAA"), i=i, pf=pf)
    assert "max_positions" in [v.rule for v in result.blocking]


def test_max_positions_does_not_block_an_add_to_a_held_symbol() -> None:
    """The position count does not change, so the limit is not engaged."""
    i = intent(max_positions=1)
    pf = book(95_000.0, AAA=(50, PRICE))
    result = check(plan_for(10, symbol="AAA"), i=i, pf=pf)
    assert "max_positions" not in [v.rule for v in result.violations]


def test_universe_membership_blocks_a_symbol_the_desk_may_not_trade() -> None:
    result = check(plan_for(10, symbol="ZZZ"), pf=book(100_000.0))
    violation = next(v for v in result.blocking if v.rule == "universe_membership")
    assert "regardless of the quality of the case" in violation.message


def test_max_decisions_per_day_blocks_past_the_cadence_limit() -> None:
    result = check(plan_for(10), prior_decisions_today=3)
    assert "max_decisions_per_day" in [v.rule for v in result.blocking]


def test_the_cadence_limit_does_not_fire_below_it() -> None:
    result = check(plan_for(10), prior_decisions_today=2)
    assert "max_decisions_per_day" not in [v.rule for v in result.violations]


def test_adv_participation_blocks_a_fill_you_could_not_get() -> None:
    result = check(plan_for(100), dollar_adv=100_000.0)  # 10,000 of 100,000
    violation = next(v for v in result.blocking if v.rule == "adv_participation")
    assert violation.actual == pytest.approx(10.0)
    assert "fiction" in violation.message


def test_adv_participation_warns_rather_than_blocks_when_unknown() -> None:
    """An unmeasurable liquidity check is a finding, not a veto -- but it is
    also not a pass, so it is recorded."""
    result = check(plan_for(10), dollar_adv=None)
    violation = next(v for v in result.violations if v.rule == "adv_participation")
    assert not violation.blocks
    assert not result.blocked


def test_a_short_is_blocked_when_shorts_are_not_allowed() -> None:
    pf = book(90_000.0, AAA=(50, PRICE))
    result = check(plan_for(-100, action="SELL"), pf=pf)
    assert "no_shorts" in [v.rule for v in result.blocking]


def test_a_stale_book_blocks() -> None:
    """Sizing against last week's weights is how a position gets doubled: the
    drift table says there is room, and there is not."""
    result = rules.check(
        plan_for(10), decision(), intent(), book(100_000.0),
        as_of=AS_OF + timedelta(days=30), sectors=SECTORS, dollar_adv=1e12,
    )
    assert "portfolio_stale" in [v.rule for v in result.blocking]


def test_a_zero_equity_book_is_rejected_outright() -> None:
    pf = PortfolioSnapshot(as_of=AS_OF, cash=0.0, positions=[])
    result = check(plan_for(10), pf=pf)
    assert "book_unusable" in [v.rule for v in result.blocking]


# --------------------------------------------------------------------------- #
# The earnings blackout -- the one rule that warns
# --------------------------------------------------------------------------- #


def estimate(basis="estimated", when=AS_OF) -> EarningsEstimate:
    return EarningsEstimate(
        symbol="AAA", next_report=when, basis=basis, source="test",
        note="a note", last_report=when - timedelta(days=91), interval_days=91,
        observations=8,
    )


def test_an_estimated_earnings_date_warns_and_does_not_block() -> None:
    """With no earnings calendar its input is a projection good to a week or
    two. Blocking on it would veto a month of every quarter on a guess."""
    result = check(plan_for(10), earnings=estimate("estimated"))
    violation = next(v for v in result.violations if v.rule == "earnings_blackout")
    assert not violation.blocks
    assert not result.blocked
    assert result.quantity == 10


def test_a_confirmed_earnings_date_blocks() -> None:
    """The flip is one field, the day a real calendar is wired up."""
    result = check(plan_for(10), earnings=estimate("confirmed"))
    violation = next(v for v in result.violations if v.rule == "earnings_blackout")
    assert violation.blocks
    assert result.blocked


def test_an_unevaluable_earnings_check_is_reported_not_passed() -> None:
    unknown = EarningsEstimate(
        symbol="AAA", next_report=None, basis="unknown", source="edgar",
        note="only annual filings",
    )
    result = check(plan_for(10), earnings=unknown)
    violation = next(v for v in result.violations if v.rule == "earnings_blackout")
    assert not violation.blocks
    assert "could not be evaluated" in violation.message


def test_an_earnings_date_outside_the_window_does_not_fire() -> None:
    far = estimate("confirmed", when=AS_OF + timedelta(days=60))
    result = check(plan_for(10), earnings=far)
    assert "earnings_blackout" not in [v.rule for v in result.violations]


# --------------------------------------------------------------------------- #
# The invariant the layer exists to guarantee
# --------------------------------------------------------------------------- #


def test_a_blocked_plan_carries_zero_shares() -> None:
    """`execute` reads ``quantity``. A blocked plan holding shares is the single
    most dangerous object this layer could produce."""
    result = check(plan_for(500), pf=book(100_000.0))
    assert result.blocked
    assert result.quantity == 0
    assert result.estimated_notional == 0.0
    assert not result.actionable


def test_the_schema_itself_refuses_a_blocked_plan_with_shares() -> None:
    """Belt and braces: no construction path can express it, not just this one."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="blocked order is zero shares"):
        OrderPlan(
            symbol="AAA", as_of=AS_OF, action="BUY", quantity=10,
            violations=[Violation(rule="max_position_pct", message="too big")],
        )


def test_the_veto_message_says_it_overrode_the_fund_manager() -> None:
    result = check(plan_for(500), pf=book(100_000.0))
    assert "the fund manager approved this order and Python vetoed it" in \
        result.skip_reason


def test_a_compliant_order_passes_with_no_violations() -> None:
    """Guard against the suite passing because everything is blocked."""
    result = check(plan_for(50), pf=book(100_000.0))
    assert not result.blocked
    assert result.actionable
    assert result.quantity == 50


def test_book_level_rules_are_recorded_even_for_a_hold() -> None:
    """"No order was placed" and "no order was placed because the book is three
    weeks stale" are different facts, and the second needs fixing."""
    hold = OrderPlan(symbol="AAA", as_of=AS_OF, action="HOLD", quantity=0)
    result = rules.check(
        hold, decision(action="HOLD"), intent(), book(100_000.0),
        as_of=AS_OF + timedelta(days=30), sectors=SECTORS,
    )
    assert "portfolio_stale" in [v.rule for v in result.violations]


# --------------------------------------------------------------------------- #
# The cadence count
# --------------------------------------------------------------------------- #


def _journal(tmp_path, *rows: tuple[str, str]):
    """Write a decision journal from ``(symbol, action)`` pairs."""
    import json

    path = tmp_path / f"decisions_{AS_OF:%Y%m%d}.jsonl"
    path.write_text("\n".join(
        json.dumps({"symbol": s, "final_decision": {"symbol": s, "action": a}})
        for s, a in rows
    ))
    return path


def test_decisions_today_counts_actionable_decisions_only(tmp_path) -> None:
    """A day of HOLDs has churned nothing, so it must not consume the budget --
    being refused your one real trade after ten HOLDs would be the limit working
    exactly backwards."""
    _journal(tmp_path, ("AAA", "HOLD"), ("BBB", "BUY"), ("CCC", "HOLD"),
             ("DDD", "SELL"))
    assert rules.decisions_today(tmp_path, AS_OF) == 2


def test_decisions_today_counts_distinct_symbols_not_runs(tmp_path) -> None:
    """Found on the first live Stage 6 run, which was vetoed with "8 actionable
    decisions already written" -- three of which were re-runs of the symbol it
    was deciding. Counting runs makes iterating on one name burn the day's
    budget, against the plan's third stopping rule."""
    _journal(tmp_path, ("AAA", "BUY"), ("AAA", "BUY"), ("AAA", "SELL"),
             ("BBB", "BUY"))
    assert rules.decisions_today(tmp_path, AS_OF) == 2


def test_decisions_today_never_counts_the_symbol_being_decided(tmp_path) -> None:
    """A re-run supersedes its own earlier proposal rather than adding a
    decision to the day. What the limit protects is how many *positions* the
    desk touches."""
    _journal(tmp_path, ("AAA", "BUY"), ("BBB", "BUY"), ("CCC", "BUY"))
    assert rules.decisions_today(tmp_path, AS_OF, exclude_symbol="AAA") == 2
    assert rules.decisions_today(tmp_path, AS_OF, exclude_symbol="ZZZ") == 3


def test_decisions_today_is_zero_when_no_journal_exists(tmp_path) -> None:
    assert rules.decisions_today(tmp_path, AS_OF) == 0


def test_decisions_today_survives_a_truncated_last_line(tmp_path) -> None:
    """Normal if a run was killed mid-write. Discarding the whole count would
    silently reset the cadence limit."""
    import json

    path = tmp_path / f"decisions_{AS_OF:%Y%m%d}.jsonl"
    path.write_text(
        json.dumps({"symbol": "AAA", "final_decision":
                    {"symbol": "AAA", "action": "BUY"}})
        + "\n{\"final_dec"
    )
    assert rules.decisions_today(tmp_path, AS_OF) == 1
