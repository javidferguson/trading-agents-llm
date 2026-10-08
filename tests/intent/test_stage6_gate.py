"""**The Stage 6 exit gate**, as the migration plan words it.

> *"**Exit gate:** an ``OrderPlan`` is produced with **zero IB contact**, and a
> deliberately non-compliant ``FinalDecision`` is blocked by Python regardless
> of what the fund manager decided."*

Two claims, two halves of this file. The second half matters more than it looks:
"regardless of what the fund manager decided" means the veto cannot be a
function of any model output, so the test drives it with a verdict that is
maximally confident, explicitly ``approve``, and wrong.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from datetime import date, datetime

import pytest

from research_desk.intent import compliance as rules
from research_desk.intent.engine import (
    compute_gaps,
    load_intent,
    load_portfolio,
    sector_map,
)
from research_desk.intent.sizing import size_order
from research_desk.models.state import FinalDecision, RiskVerdict, _final_from_verdict

AS_OF = date(2026, 10, 6)


def _decision(**overrides) -> FinalDecision:
    body = {
        "action": "BUY", "symbol": "TSM", "conviction": 0.8,
        "target_weight_pct": 7.0, "horizon_days": 60,
        "rationale": "a full sentence of reasoning goes here",
        "dissent": "the surviving argument against it goes here",
        "invalidation": "the specific condition that would prove it wrong",
        "stop_loss_pct": 8.0, "expires_at": datetime(2026, 10, 7, 12, 0),
    }
    body.update(overrides)
    return FinalDecision(**body)


def _run(decision: FinalDecision, *, dollar_adv: float | None = 2e9,
         prior: int = 0, portfolio=None):
    """The whole Stage 6 path: load, size, check. Nothing else.

    ``portfolio`` is the SEEDED book, passed in by the caller. It is not read
    from data/portfolio.yaml: that file is state the broker owns from Stage 7
    on, and a gate asserting against it would break every time the account
    moved. See tests/intent/conftest.py.
    """
    intent = load_intent()
    if portfolio is None:
        portfolio = load_portfolio()
    drift = compute_gaps(intent, portfolio, as_of=portfolio.as_of)
    sized = size_order(decision, intent, portfolio, drift,
                       as_of=portfolio.as_of,
                       reference_price=portfolio.position_value(decision.symbol)
                       and portfolio.get(decision.symbol).last_price or None)
    return rules.check(
        sized.plan, decision, intent, portfolio, as_of=portfolio.as_of,
        sectors=sector_map(), dollar_adv=dollar_adv, earnings=None,
        prior_decisions_today=prior,
    ), sized


# --------------------------------------------------------------------------- #
# Half one: an OrderPlan, with zero IB contact
# --------------------------------------------------------------------------- #


def test_an_order_plan_is_produced_from_the_seeded_book(seeded_book) -> None:
    plan, sized = _run(_decision(), portfolio=seeded_book)
    assert plan.symbol == "TSM"
    assert plan.quantity > 0
    assert plan.actionable
    assert plan.binding_cap
    assert set(plan.caps_pct) == {
        "requested", "cap_position", "cap_risk", "cap_intent"
    }


def test_the_plan_is_produced_with_no_network_access_at_all(seeded_book) -> None:
    """**Zero IB contact**, asserted by making every socket connection fail.

    Stronger than checking for ``ib_async``: it proves the path reaches no
    network of any kind, so it cannot be reading a quote, a Gateway or an API
    behind the scenes. The earnings estimate is the one piece that *may* touch
    EDGAR, and it is passed in as ``None`` here and tested separately.
    """
    original = socket.socket.connect

    def refuse(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError(
            "the Stage 6 sizing and compliance path opened a socket. It must "
            "reach no broker and no API: §9's gate is an OrderPlan produced "
            "with zero IB contact."
        )

    socket.socket.connect = refuse
    try:
        plan, _ = _run(_decision(), portfolio=seeded_book)
    finally:
        socket.socket.connect = original

    assert plan.quantity > 0


def test_the_decide_path_still_does_not_import_ib_async() -> None:
    """The §0 split, re-asserted over the Stage 6 modules specifically.

    ``tests/test_layering.py`` covers the modules that existed before; a new
    package is a new chance to reach ``execution/`` for a convenience, and
    sizing is exactly where somebody would want a live quote.
    """
    probe = textwrap.dedent("""
        import sys
        import research_desk.intent.compliance
        import research_desk.intent.earnings
        import research_desk.intent.engine
        import research_desk.intent.sizing
        import research_desk.graph.nodes.compliance
        import research_desk.models.orders
        import research_desk.models.portfolio
        leaked = [m for m in sys.modules if m.split('.')[0] == 'ib_async']
        print(','.join(leaked))
    """)
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert not [m for m in result.stdout.strip().split(",") if m]


def test_a_plan_is_produced_even_when_no_order_goes_out(seeded_book) -> None:
    """A run that wrote nothing is indistinguishable from a run that never
    happened, and the reason is the most useful field in the file."""
    plan, _ = _run(_decision(action="HOLD"), portfolio=seeded_book)
    assert plan.action == "HOLD"
    assert plan.quantity == 0
    assert plan.skip_reason


# --------------------------------------------------------------------------- #
# Half two: blocked by Python regardless of what the fund manager decided
# --------------------------------------------------------------------------- #


def _approving_verdict(**overrides) -> RiskVerdict:
    """A fund manager that could not be more certain, or more wrong."""
    body = {
        "decision": "approve", "action": "BUY", "conviction": 1.0,
        "target_weight_pct": 100.0, "horizon_days": 60,
        "rationale": "I am entirely confident in this allocation decision",
        "dissent": "there is no surviving argument against this position",
        "invalidation": "nothing could plausibly invalidate this thesis",
        "adjustment": "",
    }
    body.update(overrides)
    return RiskVerdict(**body)


def test_a_fund_manager_approving_the_whole_book_is_vetoed(seeded_book) -> None:
    """The headline case: ``approve``, conviction 1.0, 100% of equity."""
    decision = _final_from_verdict(
        _approving_verdict(), "TSM",
        expires_at=datetime(2026, 10, 7, 12, 0), stop_loss_pct=8.0,
    )
    plan, sized = _run(decision, portfolio=seeded_book)

    # Sizing alone already refuses to express it: the minimum of four caps
    # cannot exceed the smallest one.
    assert sized.target_weight_pct <= load_intent().risk.max_position_pct
    assert plan.quantity * (plan.reference_price or 0) < seeded_book.equity


@pytest.mark.parametrize(
    ("symbol", "rule"),
    [
        # Not in universe.include -- the desk may not trade it at all.
        ("GME", "universe_membership"),
    ],
)
def test_a_non_compliant_decision_is_blocked_by_rule(
    symbol: str, rule: str, seeded_book
) -> None:
    decision = _final_from_verdict(
        _approving_verdict(target_weight_pct=5.0), symbol,
        expires_at=datetime(2026, 10, 7, 12, 0), stop_loss_pct=8.0,
    )
    plan, _ = _run(decision, portfolio=seeded_book)
    assert rule in [v.rule for v in plan.blocking]
    assert plan.quantity == 0


def test_the_veto_survives_a_hand_built_plan_that_bypasses_sizing(seeded_book) -> None:
    """The veto must not depend on sizing having behaved.

    Sizing is the layer that *usually* prevents an over-sized order, so a test
    that only ever reaches compliance through sizing cannot distinguish "the
    veto works" from "sizing made the veto unnecessary".
    """
    from research_desk.models.orders import OrderPlan

    intent = load_intent()
    portfolio = seeded_book
    held = portfolio.get("TSM")

    # 2,000 shares of TSM is far above every cap, and nothing sized it.
    oversized = OrderPlan(
        symbol="TSM", as_of=portfolio.as_of, action="BUY", quantity=2_000,
        reference_price=held.last_price,
        estimated_notional=2_000 * held.last_price,
    )
    plan = rules.check(
        oversized, _decision(), intent, portfolio, as_of=portfolio.as_of,
        sectors=sector_map(), dollar_adv=2e9,
    )
    blocked = [v.rule for v in plan.blocking]
    assert "max_position_pct" in blocked
    assert "min_cash_pct" in blocked
    assert plan.quantity == 0
    assert not plan.actionable


def test_the_veto_blocks_on_the_seeded_book_at_max_positions(seeded_book) -> None:
    """The seeded book holds exactly ``max_positions``, so a new symbol is
    refused -- which is the condition the gate's drift table is built around."""
    intent = load_intent()
    portfolio = seeded_book
    assert portfolio.position_count == intent.risk.max_positions, (
        "the seeded book no longer sits at max_positions, so this gate no "
        "longer tests the condition it was built for -- fix SEED_BOOK in "
        "scripts/seed_portfolio.py, which is the fixture"
    )

    from research_desk.models.orders import OrderPlan

    # Pick the unheld symbol from the config rather than naming one. This was
    # hardcoded to GOOGL until the 2026-10-07 widening put GOOGL in the book,
    # at which point the test asserted a veto on a symbol that could not
    # trigger it. Derived, it cannot go stale again.
    unheld = next(
        s for s in intent.universe.tradeable
        if s not in portfolio.symbols and intent.theme_for(s) is not None
    )

    plan = rules.check(
        OrderPlan(symbol=unheld, as_of=portfolio.as_of, action="BUY",
                  quantity=10, reference_price=100.0,
                  estimated_notional=1_000.0),
        _decision(symbol=unheld), intent, portfolio, as_of=portfolio.as_of,
        sectors=sector_map(), dollar_adv=2e9,
    )
    assert "max_positions" in [v.rule for v in plan.blocking], (
        f"{unheld} is unheld and the book is full, so opening it must be "
        f"blocked. Got: {[v.rule for v in plan.blocking]}"
    )


def test_no_model_output_can_switch_a_check_off(seeded_book) -> None:
    """The two-key property, stated as a test.

    ``RiskVerdict.decision`` is read for reporting and is never a condition. If
    a future refactor made ``approve`` skip a check, this fails.
    """
    from research_desk.models.orders import OrderPlan

    intent = load_intent()
    portfolio = seeded_book
    held = portfolio.get("TSM")
    oversized = OrderPlan(
        symbol="TSM", as_of=portfolio.as_of, action="BUY", quantity=2_000,
        reference_price=held.last_price,
        estimated_notional=2_000 * held.last_price,
    )

    blocked_by = {}
    for verdict in ("approve", "adjust"):
        decision = _final_from_verdict(
            _approving_verdict(decision=verdict, target_weight_pct=10.0,
                               adjustment="trimmed it" if verdict == "adjust" else ""),
            "TSM", expires_at=datetime(2026, 10, 7, 12, 0), stop_loss_pct=8.0,
        )
        plan = rules.check(
            oversized, decision, intent, portfolio, as_of=portfolio.as_of,
            sectors=sector_map(), dollar_adv=2e9,
        )
        blocked_by[verdict] = sorted(v.rule for v in plan.blocking)

    assert blocked_by["approve"] == blocked_by["adjust"]
    assert blocked_by["approve"], "nothing was blocked, so this proves nothing"
