"""§9's sizing: the **minimum of four caps**, then the clamps.

Every cap gets a test in which it is the binding one, because a cap that is
never binding is untested regardless of how many other tests pass through the
function.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from research_desk.intent.engine import compute_gaps
from research_desk.intent.sizing import BUYING_POWER_FRACTION, size_order
from research_desk.models.intent import PortfolioIntent
from research_desk.models.portfolio import PortfolioSnapshot, Position
from research_desk.models.state import FinalDecision

AS_OF = date(2026, 10, 6)
PRICE = 100.0


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
        "universe": {"include": ["AAA", "BBB"], "exclude": []},
        "themes": [{"name": "T1", "target_weight_pct": 16.0, "conviction": 0.8,
                    "exemplars": ["AAA", "BBB"]}],
    })


def book(cash: float, **holdings: tuple[float, float]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=AS_OF, cash=cash,
        positions=[
            Position(symbol=s, quantity=q, avg_cost=p, last_price=p, marked_on=AS_OF)
            for s, (q, p) in holdings.items()
        ],
    )


def decision(action="BUY", weight=10.0, conviction=1.0, symbol="AAA",
             stop=8.0) -> FinalDecision:
    return FinalDecision(
        action=action, symbol=symbol, conviction=conviction,
        target_weight_pct=weight, horizon_days=60,
        rationale="a full sentence of reasoning goes here",
        dissent="the surviving argument against it goes here",
        invalidation="the specific condition that would prove it wrong",
        stop_loss_pct=stop, expires_at=datetime(2026, 10, 7, 12, 0),
    )


def size(dec: FinalDecision, i: PortfolioIntent, pf: PortfolioSnapshot,
         price: float | None = PRICE):
    drift = compute_gaps(i, pf, as_of=AS_OF)
    return size_order(dec, i, pf, drift, as_of=AS_OF, reference_price=price)


# --------------------------------------------------------------------------- #
# Each of the four caps, binding
# --------------------------------------------------------------------------- #


def test_the_requested_weight_binds_when_it_is_the_smallest_term() -> None:
    """Nothing capped it, so the approved weight stands.

    This is the common case and it is why ``requested`` belongs in the min():
    a cap set that could never be the answer would make ``binding_cap``
    misleading about why an order is the size it is.
    """
    result = size(decision(weight=4.0, conviction=0.5), intent(), book(100_000.0))
    assert result.binding_cap == "requested"
    assert result.caps_pct["requested"] == pytest.approx(4.0)
    #  4% of 100,000 = 4,000 at 100/share.
    assert result.plan.quantity == 40


@pytest.mark.parametrize("conviction", [0.01, 0.2, 0.5, 0.65, 0.95, 1.0])
def test_conviction_has_no_effect_on_the_share_count(conviction: float) -> None:
    """**Conviction is measured, never acted on.** The guard for the whole fix.

    §9's formula multiplied the target by conviction. Removing that was not
    primarily about the double-damping the first live run exposed -- it was
    that ``prompts/trader.md`` promises conviction *"will be measured against
    realised outcomes"*, and §11 makes the calibration plot (realized hit rate
    bucketed by stated confidence) a primary diagnostic. A number under
    measurement must not also be a control input, or the measurement is of a
    gamed quantity.

    Behavioural rather than a check on the cap's name, so it survives a rename
    and fails if conviction ever creeps back into sizing by another route.
    """
    result = size(decision(weight=4.0, conviction=conviction), intent(),
                  book(100_000.0))
    assert result.plan.quantity == 40
    assert result.binding_cap == "requested"


def test_the_live_tsm_decision_sizes_to_the_approved_weight(seeded_book) -> None:
    """Pinned to the run that produced this change, against the real book.

    The first live Stage 6 run: the drift table offered TSM 5.83%, the trader
    proposed 5.83% at 0.75, the fund manager **adjusted down to 4.5%** for
    overbought risk at 0.65. With ``conviction_w`` that sized to
    ``4.5 x 0.65 = 2.93%`` and ordered **+3 shares** -- the risk reduction
    applied twice, once as a judgement and once by the formula. It is +11 now,
    and the approved 4.5% is honoured.
    """
    from research_desk.intent.engine import load_intent

    real_intent = load_intent()
    # The SEEDED book, not config/portfolio.yaml: that file is state the broker
    # owns from Stage 7 on. See tests/intent/conftest.py.
    real_book = seeded_book
    drift = compute_gaps(real_intent, real_book, as_of=real_book.as_of)
    held = real_book.get("TSM")

    result = size_order(
        decision(symbol="TSM", weight=4.5, conviction=0.65),
        real_intent, real_book, drift,
        as_of=real_book.as_of, reference_price=held.last_price,
    )
    assert result.binding_cap == "requested"
    # 12 rather than the 11 first recorded here: the 2026-10-07 widening
    # re-marked the book and re-seeded it, so TSM's starting weight and price
    # both moved. The INVARIANT under test is unchanged and is what matters --
    # the approved 4.5% binds, rather than 4.5 x 0.65 = 2.93% ordering +3.
    assert result.plan.quantity == 12
    assert result.caps_pct["requested"] == pytest.approx(4.5)
    # The term that used to bind no longer exists.
    assert "conviction_w" not in result.caps_pct
    # And the conviction-scaled target would still be materially smaller, which
    # is the whole point of the fix.
    assert result.target_weight_pct > 4.5 * 0.65


def test_cap_position_binds_when_the_request_exceeds_the_ceiling() -> None:
    i = intent(max_position_pct=5.0)
    # cap_intent is 8% (16% theme / 2), cap_risk 9.375%, so cap_position wins.
    result = size(decision(weight=100.0, conviction=1.0), i, book(100_000.0))
    assert result.binding_cap == "cap_position"
    assert result.plan.quantity == 50


def test_cap_risk_binds_when_the_stop_is_wide() -> None:
    """A wider stop permits a smaller position for the same risk budget.

    0.75% of equity at risk through a 40% stop is a 1.875% position.
    """
    i = intent(max_position_pct=50.0)
    result = size(decision(weight=100.0, conviction=1.0, stop=40.0), i,
                  book(100_000.0))
    assert result.caps_pct["cap_risk"] == pytest.approx(1.875)
    assert result.binding_cap == "cap_risk"
    assert result.plan.quantity == 18  # floor(1875 / 100)


def test_a_tighter_stop_permits_a_larger_position() -> None:
    """The reason risk is expressed as risk rather than as a weight."""
    i = intent(max_position_pct=50.0)
    wide = size(decision(weight=100.0, stop=20.0), i, book(100_000.0))
    tight = size(decision(weight=100.0, stop=5.0), i, book(100_000.0))
    assert tight.caps_pct["cap_risk"] > wide.caps_pct["cap_risk"]


def test_cap_intent_binds_so_an_order_cannot_exceed_the_drift_gap() -> None:
    """The fourth cap, and the one that makes channel 2 binding rather than advisory.

    Without it the drift table says "you are 8pp light" and the model is still
    free to ask for 100%, which is exactly the "inventing allocations" §8 says
    the table exists to stop.
    """
    i = intent(max_position_pct=50.0)
    result = size(decision(weight=100.0, conviction=1.0), i, book(100_000.0))
    assert result.binding_cap == "cap_intent"
    assert result.caps_pct["cap_intent"] == pytest.approx(8.0)
    assert result.plan.quantity == 80


def test_a_symbol_with_no_mandate_cannot_be_bought() -> None:
    """``cap_intent`` is zero for a symbol in no theme, so the minimum is zero."""
    i = PortfolioIntent.from_dict({
        "objective": {"horizon_days": 60},
        "risk": intent().risk.model_dump(),
        "universe": {"include": ["AAA", "ZZZ"], "exclude": []},
        "themes": [{"name": "T1", "target_weight_pct": 16.0, "conviction": 0.8,
                    "exemplars": ["AAA"]}],
    })
    result = size(decision(symbol="ZZZ", weight=10.0), i, book(100_000.0))
    assert result.plan.quantity == 0
    assert result.caps_pct["cap_intent"] == 0.0


def test_every_cap_is_reported_even_when_it_does_not_bind() -> None:
    """"Why is this order so small" has to have a one-line answer."""
    result = size(decision(), intent(), book(100_000.0))
    assert set(result.caps_pct) == {
        "requested", "cap_position", "cap_risk", "cap_intent"
    }
    assert result.plan.caps_pct == result.caps_pct


# --------------------------------------------------------------------------- #
# The delta is against the current position, not the whole target
# --------------------------------------------------------------------------- #


def test_the_order_closes_the_gap_rather_than_buying_the_whole_target() -> None:
    #  Holding 40 shares (4%), target 8% -> buy 4% more, not 8%.
    pf = book(96_000.0, AAA=(40, PRICE))
    result = size(decision(weight=100.0, conviction=1.0), intent(), pf)
    assert result.plan.quantity == 40
    assert result.plan.current_weight_pct == pytest.approx(4.0)


def test_a_buy_with_no_room_is_refused_rather_than_turned_into_a_sell() -> None:
    """The fund manager asked to ADD. Quietly trimming would execute a trade
    nobody approved."""
    pf = book(90_000.0, AAA=(100, PRICE))  # 10%, above the 8% capped target
    result = size(decision(action="BUY", weight=100.0), intent(), pf)
    assert result.plan.quantity == 0
    assert "no room to add" in result.plan.skip_reason
    assert result.plan.action == "BUY"  # not rewritten to SELL


# --------------------------------------------------------------------------- #
# Reducing orders: the caps must not make a trim larger
# --------------------------------------------------------------------------- #


def test_a_sell_takes_the_requested_target_and_ignores_the_risk_caps() -> None:
    """A cap must never make a risk-REDUCING trade larger.

    Holding 10% and approved to trim to 5%, in a symbol whose ``cap_intent`` is
    8%. Applying the caps would take ``min(5, 8, ...)`` -- fine here -- but a
    symbol capped below the approved target would be trimmed past it and sell
    more than the fund manager allowed. So a reduction takes its requested
    target as given.
    """
    pf = book(90_000.0, AAA=(100, PRICE))
    result = size(decision(action="SELL", weight=5.0, conviction=0.5), intent(), pf)
    assert result.plan.quantity == -50
    assert "reducing" in result.binding_cap


def test_a_cap_below_the_approved_target_cannot_deepen_a_trim() -> None:
    """The case the rule above exists for, made explicit.

    Holding 10%, approved to trim to 5%, but ``cap_intent`` is 3.12%. If the
    caps applied to reductions, ``min()`` would sell down to 3.12% -- nearly
    twice the approved reduction, in the name of limiting risk.
    """
    i = PortfolioIntent.from_dict({
        "objective": {"horizon_days": 60},
        "risk": intent().risk.model_dump(),
        "universe": {"include": ["AAA", "BBB"], "exclude": []},
        # 6.25% across two exemplars -> a 3.125% per-symbol target.
        "themes": [{"name": "T1", "target_weight_pct": 6.25, "conviction": 0.5,
                    "exemplars": ["AAA", "BBB"]}],
    })
    pf = book(90_000.0, AAA=(100, PRICE))
    result = size(decision(action="SELL", weight=5.0), i, pf)
    assert result.caps_pct["cap_intent"] == pytest.approx(3.125)
    #  10% -> 5% is 50 shares. A capped trim would be 69.
    assert result.plan.quantity == -50


def test_a_sell_to_zero_closes_the_position_exactly() -> None:
    """Exactly flat, not one share short of it because of rounding."""
    pf = book(90_000.0, AAA=(100, 99.97))
    result = size(decision(action="SELL", weight=0.0), intent(), pf, price=99.97)
    assert result.plan.quantity == -100


def test_a_full_close_is_exempt_from_the_dust_floor() -> None:
    """Refusing to exit a 300 USD position would strand it in the book
    permanently, holding a max_positions slot it does not deserve."""
    pf = book(99_700.0, AAA=(3, PRICE))  # 300 USD, below min_trade_usd of 500
    result = size(decision(action="SELL", weight=0.0), intent(), pf)
    assert result.plan.quantity == -3
    assert not result.plan.skip_reason


def test_a_sell_cannot_raise_the_target() -> None:
    pf = book(97_000.0, AAA=(30, PRICE))  # 3%
    result = size(decision(action="SELL", weight=8.0), intent(), pf)
    assert result.plan.quantity == 0
    assert "the action is BUY" in result.plan.skip_reason


def test_a_sell_in_an_unheld_symbol_is_refused() -> None:
    result = size(decision(action="SELL", weight=0.0), intent(), book(100_000.0))
    assert result.plan.quantity == 0
    assert "not held" in result.plan.skip_reason


# --------------------------------------------------------------------------- #
# Dust, clamps and rounding
# --------------------------------------------------------------------------- #


def test_a_trade_below_min_trade_usd_is_rejected_as_dust() -> None:
    i = intent(min_trade_usd=5_000)
    result = size(decision(weight=2.0, conviction=1.0), i, book(100_000.0))
    assert result.plan.quantity == 0
    assert "min_trade_usd" in result.plan.skip_reason


def test_shares_are_rounded_toward_zero_never_up() -> None:
    """Rounding up would put the position above the cap the minimum respected."""
    #  8% of 100,000 = 8,000 at 199.99 -> 40.002 shares.
    result = size(decision(weight=100.0, conviction=1.0), intent(),
                  book(100_000.0), price=199.99)
    assert result.plan.quantity == 40
    assert result.plan.quantity * 199.99 <= 8_000.0


def test_max_order_shares_clamps_the_quantity() -> None:
    i = intent(max_order_shares=10)
    result = size(decision(weight=100.0, conviction=1.0), i, book(100_000.0))
    assert result.plan.quantity == 10
    assert "max_order_shares" in result.plan.binding_cap


def test_the_order_is_clamped_to_ninety_percent_of_cash() -> None:
    """§9's buying-power clamp. The 10% held back covers the gap between a
    stale mark and the price a marketable limit actually fills at."""
    #  Equity 100,000 but only 1,000 cash: an 8% target wants 8,000.
    pf = book(1_000.0, BBB=(990, PRICE))
    result = size(decision(weight=100.0, conviction=1.0), intent(), pf)
    assert result.plan.quantity == int(1_000.0 * BUYING_POWER_FRACTION // PRICE)
    assert result.plan.quantity == 9
    assert "cash" in result.plan.binding_cap


def test_cash_that_does_not_cover_one_share_produces_no_order() -> None:
    pf = book(50.0, BBB=(999, PRICE))
    result = size(decision(weight=100.0, conviction=1.0), intent(), pf)
    assert result.plan.quantity == 0
    assert "does not cover one share" in result.plan.skip_reason


def test_a_gap_smaller_than_one_share_produces_no_order() -> None:
    result = size(decision(weight=100.0, conviction=1.0), intent(),
                  book(100_000.0), price=20_000.0)
    assert result.plan.quantity == 0
    assert "less than one share" in result.plan.skip_reason


# --------------------------------------------------------------------------- #
# The cases that must never produce a share count
# --------------------------------------------------------------------------- #


def test_a_hold_is_never_sized() -> None:
    result = size(decision(action="HOLD"), intent(), book(100_000.0))
    assert result.plan.quantity == 0
    assert "HOLD" in result.plan.skip_reason


def test_no_reference_price_means_no_order() -> None:
    """A share count cannot be computed without a mark, and guessing one is
    worse than refusing."""
    result = size(decision(), intent(), book(100_000.0), price=None)
    assert result.plan.quantity == 0
    assert "no usable reference price" in result.plan.skip_reason


def test_the_books_stored_mark_is_the_fallback_price() -> None:
    """Which is the reason a ``Position`` carries one at all: ``decide`` builds
    a snapshot for one symbol and has no price for the others."""
    pf = book(96_000.0, AAA=(40, 123.45))
    drift = compute_gaps(intent(), pf, as_of=AS_OF)
    result = size_order(decision(weight=100.0, conviction=1.0), intent(), pf,
                        drift, as_of=AS_OF, reference_price=None)
    assert result.plan.reference_price == pytest.approx(123.45)
    assert result.plan.quantity > 0


def test_a_zero_equity_book_produces_no_order() -> None:
    pf = PortfolioSnapshot(as_of=AS_OF, cash=0.0, positions=[])
    result = size(decision(), intent(), pf)
    assert result.plan.quantity == 0
    assert "nothing to size against" in result.plan.skip_reason
