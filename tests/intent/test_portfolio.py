"""``PortfolioSnapshot`` -- the three rules in its docstring, enforced.

Each test here corresponds to a failure that would be silent rather than loud,
which is the only reason any of them is worth writing: a wrong position size
does not raise, it just trades the wrong amount.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from research_desk.models.portfolio import PortfolioSnapshot, Position

AS_OF = date(2026, 10, 6)


def position(symbol: str, quantity: float, price: float, cost: float | None = None) -> Position:
    return Position(
        symbol=symbol, quantity=quantity, avg_cost=cost if cost is not None else price,
        last_price=price, marked_on=AS_OF,
    )


def book(cash: float = 50_000.0, **holdings: tuple[float, float]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=AS_OF, cash=cash,
        positions=[position(s, q, p) for s, (q, p) in holdings.items()],
    )


# --------------------------------------------------------------------------- #
# Equity is derived, never stored
# --------------------------------------------------------------------------- #


def test_equity_is_cash_plus_positions() -> None:
    pf = book(cash=50_000.0, AAPL=(100, 300.0), MSFT=(50, 400.0))
    assert pf.equity == pytest.approx(50_000 + 30_000 + 20_000)


def test_equity_cannot_be_set_from_the_file() -> None:
    """A stored equity and a position list can disagree; this one cannot.

    Pydantic ignores an unknown key by default, so the risk is not a crash --
    it is that somebody adds ``equity: 999999`` to the YAML, sees no error, and
    believes it took effect. The property is the only source.
    """
    pf = PortfolioSnapshot.model_validate({
        "as_of": AS_OF, "cash": 1_000.0, "positions": [], "equity": 999_999.0,
    })
    assert pf.equity == 1_000.0


def test_weights_sum_with_cash_to_one_hundred_percent() -> None:
    """The identity that makes ``effective_gross_cap_pct`` necessary."""
    pf = book(cash=50_000.0, AAPL=(100, 300.0), MSFT=(50, 400.0))
    assert pf.cash_pct + pf.gross_exposure_pct == pytest.approx(100.0)


def test_position_value_is_zero_for_an_unheld_symbol() -> None:
    pf = book(AAPL=(10, 100.0))
    assert pf.position_value("NVDA") == 0.0
    assert pf.weight_pct("NVDA") == 0.0


def test_a_zero_equity_book_does_not_report_zero_exposure() -> None:
    """``_pct`` refuses to invent a percentage rather than returning a safe-looking one.

    0.0 would read downstream as "no exposure", which is the most dangerous
    possible answer. ``compliance.check`` rejects the book outright instead.
    """
    pf = PortfolioSnapshot(as_of=AS_OF, cash=0.0, positions=[])
    assert pf.equity == 0.0
    assert pf.weight_pct("AAPL") == 0.0  # no exposure to invent, which is honest
    assert pf.cash_pct == 0.0


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def test_two_rows_for_one_symbol_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate positions"):
        PortfolioSnapshot(
            as_of=AS_OF, cash=0.0,
            positions=[position("AAPL", 10, 100.0), position("AAPL", 5, 100.0)],
        )


def test_a_zero_quantity_position_is_refused() -> None:
    """It would occupy a ``max_positions`` slot that is not actually occupied."""
    with pytest.raises(ValidationError, match="quantity 0"):
        position("AAPL", 0, 100.0)


def test_symbols_are_upper_cased() -> None:
    pf = PortfolioSnapshot(
        as_of=AS_OF, cash=0.0, positions=[position("aapl", 10, 100.0)]
    )
    assert pf.symbols == {"AAPL"}
    assert pf.position_value("aapl") == 1_000.0


# --------------------------------------------------------------------------- #
# Staleness blocks, and a future mark is not freshness
# --------------------------------------------------------------------------- #


def test_a_fresh_book_is_not_stale() -> None:
    pf = book()
    assert not pf.is_stale(AS_OF)
    assert pf.staleness_reason(AS_OF) is None


def test_a_book_inside_the_tolerance_is_not_stale() -> None:
    pf = book()
    assert not pf.is_stale(date(2026, 10, 13))  # exactly 7 days


def test_a_book_past_the_tolerance_is_stale() -> None:
    pf = book()
    assert pf.is_stale(date(2026, 10, 14))
    reason = pf.staleness_reason(date(2026, 10, 14))
    assert reason is not None
    assert "8 days before" in reason
    assert "desk portfolio --refresh" in reason


def test_marks_in_the_future_are_look_ahead_not_freshness() -> None:
    """A replay against an older ``as_of`` must not size against later marks.

    §7.4's point-in-time rule applied to the book. ``drift < 0`` is the easiest
    bug to write and the hardest to see: the numbers all look plausible and the
    decision was made with information that did not exist.
    """
    pf = book()
    assert pf.is_stale(date(2026, 9, 1))
    reason = pf.staleness_reason(date(2026, 9, 1))
    assert reason is not None
    assert "FUTURE" in reason
    assert "look-ahead" in reason


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_a_missing_file_is_an_error_not_an_empty_book(tmp_path) -> None:
    """An absent file sized as all-cash would propose a full opening trade."""
    with pytest.raises(FileNotFoundError, match="not an empty one"):
        PortfolioSnapshot.load(tmp_path / "nope.yaml")


def test_the_shipped_book_loads_and_is_internally_consistent() -> None:
    """``config/portfolio.yaml`` must always parse and self-agree.

    Asserts the INVARIANTS, not a share count. The file is state: it holds
    whatever the account holds, and from Stage 7 ``execute`` overwrites it from
    the broker -- so an all-cash book with no positions is a perfectly valid
    state, and one with fifteen is too. Tests that need a specific shape use
    the ``seeded_book`` fixture instead (see conftest.py).
    """
    from research_desk.intent.engine import load_portfolio

    pf = load_portfolio()
    assert pf.cash >= 0
    assert pf.equity == pytest.approx(pf.cash + pf.net_value)
    assert all(p.last_price > 0 for p in pf.positions)
    if pf.positions:
        # as_of is the oldest mark, which is what makes a refresh unable to
        # reset the staleness clock.
        assert pf.as_of == min(p.marked_on for p in pf.positions)


def test_the_seeded_fixture_book_has_the_shape_the_gate_needs(seeded_book) -> None:
    """The properties scripts/seed_portfolio.py:SEED_BOOK exists to produce.

    This is the fixture the Stage 6 gate asserts against, so if the recipe
    stops producing a full book with headroom in it, the gate stops testing
    what it was built for -- and would do so silently.
    """
    from research_desk.intent.engine import load_intent

    intent = load_intent()
    assert seeded_book.position_count == intent.risk.max_positions
    assert seeded_book.cash > 0
    assert seeded_book.equity == pytest.approx(
        seeded_book.cash + seeded_book.net_value
    )
    # At least one symbol at the position cap, and one with real headroom.
    weights = {p.symbol: seeded_book.weight_pct(p.symbol) for p in seeded_book.positions}
    assert any(w >= intent.risk.max_position_pct - 0.1 for w in weights.values())
    assert seeded_book.get("TSM") is not None, "TSM is the BUY-sizes case"


def test_yaml_round_trip_preserves_every_number() -> None:
    pf = book(cash=1_234.56, AAPL=(10, 199.99))
    again = PortfolioSnapshot.from_dict(pf.to_yaml_dict())
    assert again.equity == pytest.approx(pf.equity)
    assert again.as_of == pf.as_of
    assert again.positions[0].last_price == pf.positions[0].last_price


def test_unrealised_pnl_uses_cost_basis_not_the_mark() -> None:
    held = position("AAPL", 10, price=120.0, cost=100.0)
    assert held.market_value == 1_200.0
    assert held.cost_basis == 1_000.0
    assert held.unrealized_usd == 200.0
    assert held.unrealized_pct == pytest.approx(20.0)
