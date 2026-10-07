"""Book vs broker. **A mismatch is a refusal, not a warning.**

The share count on a proposal was computed from ``config/portfolio.yaml``. If
the file disagrees with the account, the number is wrong, and approving it means
approving arithmetic over a bad input. So the order gate is never drawn.

`execute` is the only process that can see the real positions -- that is the §0
split working -- which also makes it the only place this check can live.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_desk.execution import reconcile
from research_desk.models.portfolio import PortfolioSnapshot, Position

from .conftest import FakeAccountValue, FakeContract, FakeIB, FakePosition

AS_OF = date(2026, 10, 7)


def book(**holdings: float) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=AS_OF, cash=90_000.0,
        positions=[
            Position(symbol=s, quantity=q, avg_cost=100.0,
                     last_price=100.0, marked_on=AS_OF)
            for s, q in holdings.items()
        ],
    )


def holding(symbol: str, quantity: float, cost: float = 100.0) -> reconcile.Holding:
    return reconcile.Holding(symbol=symbol, quantity=quantity, avg_cost=cost)


# --------------------------------------------------------------------------- #
# What counts as a match
# --------------------------------------------------------------------------- #


def test_identical_positions_match() -> None:
    result = reconcile.compare(book(TSM=11, NVDA=105),
                               {"TSM": holding("TSM", 11),
                                "NVDA": holding("NVDA", 105)})
    assert result.matches
    assert not result.mismatched


def test_a_one_share_difference_is_a_mismatch() -> None:
    """Exact, deliberately. A share count is an integer, and one share is a
    real difference rather than noise -- there is no tolerance that makes
    "approximately 105 shares" meaningful."""
    result = reconcile.compare(book(NVDA=105), {"NVDA": holding("NVDA", 104)})
    assert not result.matches
    assert result.mismatched[0].state == "QUANTITY DIFFERS"


def test_a_position_the_broker_has_and_the_book_does_not() -> None:
    result = reconcile.compare(book(TSM=11),
                               {"TSM": holding("TSM", 11),
                                "BRK.B": holding("BRK.B", 5)})
    assert not result.matches
    row = next(r for r in result.rows if r.symbol == "BRK.B")
    assert row.state == "NOT IN BOOK"


def test_a_position_the_book_has_and_the_broker_does_not() -> None:
    result = reconcile.compare(book(TSM=11, QQQ=7), {"TSM": holding("TSM", 11)})
    assert not result.matches
    row = next(r for r in result.rows if r.symbol == "QQQ")
    assert row.state == "NOT AT BROKER"


def test_cash_and_equity_do_not_gate_the_order() -> None:
    """Floats that move with every dividend, interest accrual and commission.
    Comparing them exactly would reject every run, so only positions gate."""
    result = reconcile.compare(
        book(TSM=11), {"TSM": holding("TSM", 11)},
        broker_equity=249_000.0, broker_cash=88_000.0,
    )
    assert result.matches
    assert result.equity_drift_pct is not None


def test_equity_drift_is_reported_as_context() -> None:
    result = reconcile.compare(
        book(TSM=11), {"TSM": holding("TSM", 11)}, broker_equity=200_000.0,
    )
    text = reconcile.render(result)
    assert "NOTE" in text
    assert "context" in text


def test_an_empty_book_against_an_empty_account_matches() -> None:
    result = reconcile.compare(book(), {})
    assert result.matches
    assert result.rows == ()


# --------------------------------------------------------------------------- #
# What the human is shown
# --------------------------------------------------------------------------- #


def test_the_table_shows_both_sides_even_on_a_match() -> None:
    """Worth one line of reassurance before approving an order sized from it."""
    text = reconcile.render(reconcile.compare(
        book(TSM=11), {"TSM": holding("TSM", 11)},
        broker_equity=250_000.0, broker_cash=90_000.0, broker_account="DU1234567",
    ))
    assert "BOOK vs BROKER" in text
    assert "DU1234567" in text
    assert "TSM" in text
    assert "ok" in text


def test_the_rejection_names_every_mismatch_and_the_fix() -> None:
    result = reconcile.compare(
        book(NVDA=105, QQQ=7),
        {"NVDA": holding("NVDA", 90), "BRK.B": holding("BRK.B", 5)},
    )
    text = reconcile.render_rejection(result, "config/portfolio.yaml")

    assert "REJECTED" in text
    assert "NO ORDER WAS SHOWN" in text
    for symbol in ("NVDA", "QQQ", "BRK.B"):
        assert symbol in text
    # And it must say what to do, not only that something is wrong.
    assert "desk decide" in text
    assert "config/portfolio.yaml" in text


def test_the_rejection_explains_why_approval_is_not_offered() -> None:
    """The reasoning matters: it is not that the order looks risky, it is that
    the number on screen was computed from a figure known to be wrong."""
    result = reconcile.compare(book(NVDA=105), {"NVDA": holding("NVDA", 90)})
    text = reconcile.render_rejection(result, "p.yaml")
    assert "nothing safe to approve" in text


# --------------------------------------------------------------------------- #
# Reading the account
# --------------------------------------------------------------------------- #


async def test_holdings_use_the_canonical_ticker_not_ibs_spelling() -> None:
    """IB reports BRK.B as "BRK B"; the book uses the dotted form, and a
    mismatch in spelling would read as a mismatch in position."""
    ib = FakeIB(positions=[
        FakePosition(FakeContract(symbol="BRK B"), 5, 500.0),
        FakePosition(FakeContract(symbol="TSM"), 11, 370.0),
    ])
    holdings = await reconcile.holdings_from_ib(ib)
    assert set(holdings) == {"BRK.B", "TSM"}
    assert holdings["BRK.B"].quantity == 5


async def test_non_stock_positions_are_ignored_with_a_warning(caplog) -> None:
    """An option in the account is not something this desk put there, and
    folding one into an equity book would make every weight nonsense."""
    ib = FakeIB(positions=[
        FakePosition(FakeContract(symbol="SPY", secType="OPT",
                                  localSymbol="SPY 251219C00700000"), 2, 500.0),
        FakePosition(FakeContract(symbol="TSM"), 11, 370.0),
    ])
    with caplog.at_level("WARNING"):
        holdings = await reconcile.holdings_from_ib(ib)
    assert set(holdings) == {"TSM"}
    assert any("non-stock position" in m for m in caplog.messages)


async def test_account_values_are_parsed_as_floats() -> None:
    ib = FakeIB()
    values = await reconcile.account_values(ib)
    assert values["NetLiquidation"] == pytest.approx(250_000.0)
    assert values["TotalCashValue"] == pytest.approx(90_279.72)


# --------------------------------------------------------------------------- #
# Writing the book back -- what Stage 6 promised
# --------------------------------------------------------------------------- #


async def test_the_written_book_is_marked_as_coming_from_ib() -> None:
    """``config/portfolio.yaml``'s own header has promised this since Stage 6:
    *"At Stage 7, execute overwrites this file from the live paper account and
    source: becomes ib."*"""
    ib = FakeIB(positions=[
        FakePosition(FakeContract(symbol="TSM"), 11, 370.50),
        FakePosition(FakeContract(symbol="NVDA"), 105, 201.68),
    ])
    snapshot = await reconcile.book_from_ib(ib, as_of=AS_OF)

    assert snapshot.source == "ib"
    assert snapshot.position_count == 2
    assert snapshot.get("TSM").quantity == 11
    assert snapshot.cash == pytest.approx(90_279.72)
    assert snapshot.as_of == AS_OF
    # The note must say the marks are not market prices.
    assert "average cost" in snapshot.note


async def test_a_zero_quantity_position_is_dropped() -> None:
    """IB reports closed positions as 0, and the schema rejects them -- a
    zero-quantity row would occupy a max_positions slot that is not occupied."""
    ib = FakeIB(positions=[
        FakePosition(FakeContract(symbol="TSM"), 0, 370.50),
        FakePosition(FakeContract(symbol="NVDA"), 105, 201.68),
    ])
    snapshot = await reconcile.book_from_ib(ib, as_of=AS_OF)
    assert snapshot.symbols == {"NVDA"}


async def test_a_zero_avg_cost_still_produces_a_valid_price() -> None:
    """The schema requires ``last_price > 0``. A broker that reports no average
    cost must not make the whole write fail."""
    ib = FakeIB(positions=[FakePosition(FakeContract(symbol="TSM"), 11, 0.0)])
    snapshot = await reconcile.book_from_ib(ib, as_of=AS_OF)
    assert snapshot.get("TSM").last_price > 0


async def test_a_missing_cash_figure_refuses_rather_than_guessing() -> None:
    """Equity is derived as cash plus positions, so a wrong cash figure
    silently rescales every weight in the book."""
    ib = FakeIB(
        positions=[FakePosition(FakeContract(symbol="TSM"), 11, 370.0)],
        account_values=[FakeAccountValue("NetLiquidation", "250000.00")],
    )
    with pytest.raises(RuntimeError, match="TotalCashValue"):
        await reconcile.book_from_ib(ib, as_of=AS_OF)


async def test_the_written_book_round_trips_through_yaml() -> None:
    """It is about to be written to a file that `decide` reads next run."""
    ib = FakeIB(positions=[FakePosition(FakeContract(symbol="TSM"), 11, 370.50)])
    snapshot = await reconcile.book_from_ib(ib, as_of=AS_OF)
    again = PortfolioSnapshot.from_dict(snapshot.to_yaml_dict())
    assert again.equity == pytest.approx(snapshot.equity)
    assert again.source == "ib"
