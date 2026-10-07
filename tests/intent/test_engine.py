"""``compute_gaps()`` and the pre-filter. Channels 1 and 2 (§8).

The arithmetic is the whole point -- §2 says models cannot do it, so if these
numbers are wrong the design's main reliability claim is wrong with them.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from research_desk.intent.engine import (
    BAND_FLOOR_PCT,
    band_for,
    candidates,
    compute_gaps,
    load_intent,
    load_portfolio,
    sector_map,
)
from research_desk.models.intent import PortfolioIntent
from research_desk.models.portfolio import PortfolioSnapshot, Position

AS_OF = date(2026, 10, 6)


def intent(**overrides) -> PortfolioIntent:
    body = {
        "objective": {"horizon_days": 60, "benchmark": "SPY"},
        "risk": {
            "per_trade_risk_pct": 0.75, "default_stop_pct": 8.0,
            "max_position_pct": 10.0, "max_sector_pct": 65.0,
            "min_cash_pct": 10.0, "max_gross_exposure_pct": 95.0,
            "max_positions": 8, "min_trade_usd": 500, "max_order_shares": 2000,
            "max_adv_participation_pct": 2.0,
            "earnings_blackout_days_before": 2, "earnings_blackout_days_after": 1,
        },
        "universe": {"include": ["AAA", "BBB", "CCC", "DDD"], "exclude": []},
        "themes": [
            {"name": "T1", "target_weight_pct": 20.0, "conviction": 0.8,
             "exemplars": ["AAA", "BBB"]},
            {"name": "T2", "target_weight_pct": 10.0, "conviction": 0.4,
             "exemplars": ["CCC"]},
        ],
    }
    for key, value in overrides.items():
        body[key] = value
    return PortfolioIntent.from_dict(body)


def book(cash: float, **holdings: tuple[float, float]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=AS_OF, cash=cash,
        positions=[
            Position(symbol=s, quantity=q, avg_cost=p, last_price=p, marked_on=AS_OF)
            for s, (q, p) in holdings.items()
        ],
    )


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #


def test_the_band_is_proportional_above_the_floor() -> None:
    assert band_for(20.0) == pytest.approx(5.0)   # 25% of 20
    assert band_for(35.0) == pytest.approx(8.75)


def test_the_band_never_falls_below_the_floor() -> None:
    """A fixed band would be impossible on a small target; the floor stops a
    day's price move from pushing a 2% position out of band every day."""
    assert band_for(2.0) == BAND_FLOOR_PCT
    assert band_for(0.0) == BAND_FLOOR_PCT


# --------------------------------------------------------------------------- #
# The per-symbol target: a theme total divided, then capped
# --------------------------------------------------------------------------- #


def test_a_theme_target_is_divided_across_its_exemplars() -> None:
    """The Stage 3 failure, now arithmetic instead of a prompt instruction.

    Four of ten Stage 3 proposals read a theme's 35% total as the symbol's own
    target. This is the half of the fix that a model cannot get wrong.
    """
    drift = compute_gaps(intent(), book(100_000.0), as_of=AS_OF)
    # T1 is 20% across two exemplars.
    assert drift.row("AAA").target_weight_pct == pytest.approx(10.0)
    assert drift.row("BBB").target_weight_pct == pytest.approx(10.0)
    # T2 is 10% across one.
    assert drift.row("CCC").target_weight_pct == pytest.approx(10.0)


def test_a_symbol_target_is_capped_by_max_position_pct() -> None:
    """A two-exemplar 35% theme implies 17.5% each against a 10% ceiling."""
    i = intent(themes=[{
        "name": "Concentrated", "target_weight_pct": 35.0, "conviction": 0.9,
        "exemplars": ["AAA", "BBB"],
    }])
    drift = compute_gaps(i, book(100_000.0), as_of=AS_OF)
    row = drift.row("AAA")
    assert row.target_weight_pct == pytest.approx(10.0)
    assert "max_position_pct caps" in row.target_note


def test_a_symbol_in_no_theme_has_a_zero_target_and_says_why() -> None:
    """Not an oversight: the book has no mandate for it, and sizing enforces
    that by capping the order at the target."""
    drift = compute_gaps(intent(), book(100_000.0), as_of=AS_OF)
    row = drift.row("DDD")
    assert row.target_weight_pct == 0.0
    assert row.theme is None
    assert "no mandate" in row.target_note


# --------------------------------------------------------------------------- #
# Gaps, in percent and in dollars
# --------------------------------------------------------------------------- #


def test_an_empty_book_shows_every_gap_fully_open() -> None:
    drift = compute_gaps(intent(), book(100_000.0), as_of=AS_OF)
    assert drift.row("AAA").gap_pct == pytest.approx(10.0)
    assert drift.row("AAA").gap_usd == pytest.approx(10_000.0)
    assert drift.row("AAA").direction == "underweight"


def test_an_overweight_symbol_shows_a_negative_gap() -> None:
    #  AAA: 150 x 100 = 15,000 of 100,000 equity = 15%, target 10%
    drift = compute_gaps(intent(), book(85_000.0, AAA=(150, 100.0)), as_of=AS_OF)
    row = drift.row("AAA")
    assert row.current_weight_pct == pytest.approx(15.0)
    assert row.gap_pct == pytest.approx(-5.0)
    assert row.gap_usd == pytest.approx(-5_000.0)
    assert row.direction == "overweight"


def test_a_gap_inside_the_band_reads_as_in_band() -> None:
    """What makes "or don't" a real option rather than a to-do list."""
    #  AAA at 9.5% against a 10% target, band 2.5pp.
    drift = compute_gaps(intent(), book(90_500.0, AAA=(95, 100.0)), as_of=AS_OF)
    row = drift.row("AAA")
    assert row.in_band
    assert row.direction == "in band"


def test_the_theme_row_is_the_total_across_its_symbols() -> None:
    pf = book(80_000.0, AAA=(100, 100.0), BBB=(100, 100.0))
    drift = compute_gaps(intent(), pf, as_of=AS_OF)
    row = drift.theme_row("T1")
    assert row.current_weight_pct == pytest.approx(20.0)
    assert row.gap_pct == pytest.approx(0.0)
    assert row.held == ("AAA", "BBB")
    assert row.unheld == ()


def test_an_unheld_exemplar_is_named_on_the_theme_row() -> None:
    drift = compute_gaps(intent(), book(90_000.0, AAA=(100, 100.0)), as_of=AS_OF)
    assert drift.theme_row("T1").unheld == ("BBB",)


def test_every_tradeable_symbol_gets_a_row_even_when_unheld() -> None:
    """Omitting them would make the trader infer the universe from the holdings."""
    drift = compute_gaps(intent(), book(100_000.0), as_of=AS_OF)
    assert {r.symbol for r in drift.symbols} == {"AAA", "BBB", "CCC", "DDD"}


def test_a_holding_in_no_theme_is_named_explicitly() -> None:
    """It counts toward gross exposure and max_positions, toward no target --
    so it is invisible in the theme table unless something says so."""
    drift = compute_gaps(intent(), book(90_000.0, DDD=(100, 100.0)), as_of=AS_OF)
    assert drift.unthemed_holdings == ("DDD",)


def test_the_book_level_facts_are_computed_not_left_to_the_model() -> None:
    pf = book(90_000.0, AAA=(100, 100.0))
    drift = compute_gaps(intent(), pf, as_of=AS_OF)
    assert drift.equity == pytest.approx(100_000.0)
    assert drift.cash_pct == pytest.approx(90.0)
    assert drift.gross_exposure_pct == pytest.approx(10.0)
    assert drift.position_count == 1
    assert drift.slots_free == 7


# --------------------------------------------------------------------------- #
# Channel 1 -- the pre-filter
# --------------------------------------------------------------------------- #


def test_the_prefilter_drops_an_excluded_symbol() -> None:
    i = intent(universe={"include": ["AAA", "BBB"], "exclude": ["BBB"]},
               themes=[{"name": "T1", "target_weight_pct": 20.0,
                        "conviction": 0.8, "exemplars": ["AAA"]}])
    result = candidates(i, book(100_000.0), as_of=AS_OF)
    assert result.eligible == ("AAA",)
    assert "universe.exclude" in result.reason("BBB")


def test_the_prefilter_drops_a_symbol_in_an_earnings_blackout() -> None:
    result = candidates(
        intent(), book(100_000.0), as_of=AS_OF,
        blackouts={"BBB": "inside the blackout window"},
    )
    assert "BBB" not in result.eligible
    assert result.reason("BBB") == "inside the blackout window"


def test_at_max_positions_a_new_symbol_is_dropped() -> None:
    i = intent(risk={**intent().risk.model_dump(), "max_positions": 1})
    result = candidates(i, book(90_000.0, AAA=(100, 100.0)), as_of=AS_OF)
    assert "BBB" not in result.eligible
    assert "max_positions" in result.reason("BBB")


def test_at_max_positions_a_HELD_symbol_is_still_a_candidate() -> None:
    """A full book must stay rebalanceable.

    Trimming, closing or adding to a symbol the book already holds changes no
    slot count. Filtering it out would make the first thing you want to do with
    a full book -- rebalance it -- impossible.
    """
    i = intent(risk={**intent().risk.model_dump(), "max_positions": 1})
    result = candidates(i, book(90_000.0, AAA=(100, 100.0)), as_of=AS_OF)
    assert result.admits("AAA")


def test_the_prefilter_gives_a_reason_per_symbol_not_a_count() -> None:
    """"11 of 14 filtered" is unactionable; the reason is a config review."""
    i = intent(risk={**intent().risk.model_dump(), "max_positions": 1})
    result = candidates(i, book(90_000.0, AAA=(100, 100.0)), as_of=AS_OF)
    assert set(result.rejected) == {"BBB", "CCC", "DDD"}
    assert all(result.rejected.values())


# --------------------------------------------------------------------------- #
# Intent validation: a missing cap must not read as "no limit"
# --------------------------------------------------------------------------- #


def test_a_missing_risk_limit_fails_at_load() -> None:
    body = {
        "objective": {"horizon_days": 60},
        "risk": {"per_trade_risk_pct": 0.75},
        "universe": {"include": ["AAA"]},
    }
    with pytest.raises(ValidationError):
        PortfolioIntent.from_dict(body)


def test_a_risk_cap_above_the_position_cap_is_refused() -> None:
    """It could never be the binding term, which makes it decoration."""
    risk = {**intent().risk.model_dump(), "per_trade_risk_pct": 20.0}
    with pytest.raises(ValidationError, match="decoration"):
        intent(risk=risk)


def test_the_long_only_cash_floor_caps_gross_below_the_declared_limit() -> None:
    """The pair that looks like a contradiction and is not.

    ``min_cash_pct`` 10 and ``max_gross_exposure_pct`` 95 sum to 105 in the
    shipped config. In a long-only book ``cash_pct + gross_pct == 100``, so the
    cash floor already caps gross at 90 and the 95 never binds -- redundant,
    not impossible. It starts mattering if shorts are ever allowed.
    """
    risk = intent().risk
    assert risk.effective_gross_cap_pct(allow_shorts=False) == pytest.approx(90.0)
    assert risk.effective_gross_cap_pct(allow_shorts=True) == pytest.approx(95.0)


def test_a_symbol_in_two_themes_is_refused() -> None:
    """Its weight would count toward two theme targets at once."""
    with pytest.raises(ValidationError, match="appears in both"):
        intent(themes=[
            {"name": "T1", "target_weight_pct": 10.0, "conviction": 0.5,
             "exemplars": ["AAA"]},
            {"name": "T2", "target_weight_pct": 10.0, "conviction": 0.5,
             "exemplars": ["AAA"]},
        ])


def test_an_exemplar_outside_the_universe_is_refused() -> None:
    """Its gap could never be closed, and nothing would say why."""
    with pytest.raises(ValidationError, match="not in universe.include"):
        intent(themes=[{"name": "T1", "target_weight_pct": 10.0,
                        "conviction": 0.5, "exemplars": ["ZZZ"]}])


def test_theme_targets_above_the_gross_limit_are_refused() -> None:
    with pytest.raises(ValidationError, match="more exposure than compliance allows"):
        intent(themes=[{"name": "T1", "target_weight_pct": 100.0,
                        "conviction": 0.5, "exemplars": ["AAA", "BBB"]}])


def test_the_confirmation_gate_cannot_be_turned_off_in_code() -> None:
    """The second lock. ``config.load_config`` is the first.

    The ORB engine's options scanner had an INVERTED gate: false still
    prompted, then returned True regardless of the answer. Two locks is the
    right number for that class of bug.
    """
    with pytest.raises(ValidationError):
        intent(execution={"require_confirmation": False})


# --------------------------------------------------------------------------- #
# Against the real, shipped config
# --------------------------------------------------------------------------- #


def test_the_shipped_intent_loads_as_the_typed_object() -> None:
    i = load_intent()
    assert i.risk.max_position_pct > 0
    assert i.themes
    assert i.universe.tradeable


def test_the_shipped_drift_table_is_arithmetically_consistent() -> None:
    i, pf = load_intent(), load_portfolio()
    drift = compute_gaps(i, pf, as_of=pf.as_of)

    for row in drift.symbols:
        assert row.gap_pct == pytest.approx(
            row.target_weight_pct - row.current_weight_pct
        )
        assert row.gap_usd == pytest.approx(drift.equity * row.gap_pct / 100.0)

    for row in drift.themes:
        theme = i.theme_named(row.name)
        expected = sum(pf.weight_pct(s) for s in theme.exemplars)
        assert row.current_weight_pct == pytest.approx(expected)


def test_the_sector_map_covers_every_tradeable_symbol() -> None:
    """``max_sector_pct`` cannot be enforced on a symbol with no sector."""
    sectors = sector_map()
    missing = [s for s in load_intent().universe.tradeable if s not in sectors]
    assert not missing, (
        f"{missing} have no sector_etf in universe.yaml, so max_sector_pct "
        "cannot be evaluated for them"
    )
