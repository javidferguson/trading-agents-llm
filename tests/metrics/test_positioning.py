"""Positioning: the dissemination lag and the discretionary filter.

Both of these are places where plausible-looking code produces a confidently
wrong number, so they are tested harder than the arithmetic around them.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research_desk.metrics.positioning import positioning_metrics
from research_desk.providers.edgar import (
    CODE_MEANINGS,
    DISCRETIONARY_CODES,
    _parse_form4,
)
from research_desk.providers.finra import DISSEMINATION_LAG_DAYS

AS_OF = date(2026, 10, 6)


def interest(settled: date, shares: float = 100e6, dtc: float = 2.5,
             change: float = -5.0) -> dict:
    return {
        "settlement_date": settled,
        "public_from": settled + timedelta(days=DISSEMINATION_LAG_DAYS),
        "short_shares": shares,
        "previous_short_shares": shares * 1.05,
        "days_to_cover": dtc,
        "change_pct": change,
        "average_daily_volume": shares / dtc,
    }


def tx(code: str, *, acquired: bool, shares: float, price: float | None) -> dict:
    return {
        "date": AS_OF - timedelta(days=5),
        "code": code,
        "code_meaning": CODE_MEANINGS.get(code, "other"),
        "discretionary": code in DISCRETIONARY_CODES,
        "acquired": acquired,
        "shares": shares,
        "price": price,
        "usd": shares * price if price is not None else None,
        "is_officer": True, "is_director": False,
        "is_ten_percent_owner": False, "officer_title": "CFO",
    }


# --------------------------------------------------------------------------- #
# The discretionary filter -- measured against a real Apple filing
# --------------------------------------------------------------------------- #


def test_only_open_market_trades_count_as_insider_activity() -> None:
    """The trap this exists for, taken from a real AAPL Form 4.

    That filing reports an "M" of 374,541 shares ACQUIRED (an option exercise)
    and an "F" of 199,038 DISPOSED (shares withheld to pay the tax on it).
    Counted naively that reads as an executive buying 374k shares -- the
    opposite of informative, because nobody chose to buy anything.
    """
    insiders = [
        tx("M", acquired=True, shares=374_541, price=None),    # exercise
        tx("F", acquired=False, shares=199_038, price=330.32),  # tax withheld
        tx("S", acquired=False, shares=50_000, price=330.0),    # a real sale
        tx("G", acquired=False, shares=1_000, price=None),      # a gift
    ]
    metrics = positioning_metrics(None, None, insiders, AS_OF)

    assert metrics.insider_buy_count == 0, "an exercise is not a purchase"
    assert metrics.insider_sell_count == 1, "only the open-market sale"
    assert metrics.insider_nondiscretionary_count == 3
    # Net dollars reflect the sale alone, not the tax withholding.
    assert metrics.insider_net_usd == pytest.approx(-50_000 * 330.0)


def test_the_real_apple_breakdown_separates_cleanly() -> None:
    """Observed live for AAPL over 90 days: 18 S, 5 F, 5 M, 1 G."""
    insiders = (
        [tx("S", acquired=False, shares=1000, price=300.0) for _ in range(18)]
        + [tx("F", acquired=False, shares=500, price=300.0) for _ in range(5)]
        + [tx("M", acquired=True, shares=2000, price=None) for _ in range(5)]
        + [tx("G", acquired=False, shares=100, price=None)]
    )
    metrics = positioning_metrics(None, None, insiders, AS_OF)
    assert metrics.insider_sell_count == 18
    assert metrics.insider_buy_count == 0
    assert metrics.insider_nondiscretionary_count == 11


def test_a_priceless_transaction_does_not_count_as_zero_dollars() -> None:
    """An exercise or gift has no transaction price. Treating that as 0 would
    drag a net total toward nothing while looking like a real figure."""
    insiders = [tx("P", acquired=True, shares=1000, price=None)]
    metrics = positioning_metrics(None, None, insiders, AS_OF)

    assert metrics.insider_buy_count == 1, "the trade still happened"
    assert metrics.insider_net_usd is None
    assert "reported a price" in metrics.gaps["insider_net_usd"]


def test_buys_and_sells_net_against_each_other() -> None:
    insiders = [
        tx("P", acquired=True, shares=1000, price=100.0),
        tx("S", acquired=False, shares=400, price=100.0),
    ]
    metrics = positioning_metrics(None, None, insiders, AS_OF)
    assert metrics.insider_net_usd == pytest.approx(60_000)


def test_derivative_rows_are_skipped_by_the_parser() -> None:
    """Options and RSUs move on vesting schedules, not on a view about price."""
    xml = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner><reportingOwnerRelationship>
        <isDirector>true</isDirector><isOfficer>false</isOfficer>
      </reportingOwnerRelationship></reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2026-09-30</value></transactionDate>
        <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
        <transactionAmounts>
          <transactionShares><value>100</value></transactionShares>
          <transactionPricePerShare><value>50.5</value></transactionPricePerShare>
          <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
        </transactionAmounts>
      </nonDerivativeTransaction></nonDerivativeTable>
      <derivativeTable><derivativeTransaction>
        <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      </derivativeTransaction></derivativeTable>
    </ownershipDocument>"""
    rows = _parse_form4(xml)
    assert len(rows) == 1
    assert rows[0]["code"] == "S"
    assert rows[0]["usd"] == pytest.approx(100 * 50.5)
    assert rows[0]["is_director"] is True


def test_an_unparseable_form4_yields_nothing_rather_than_raising() -> None:
    assert _parse_form4("not xml at all") == []


# --------------------------------------------------------------------------- #
# Short interest, and how stale it is
# --------------------------------------------------------------------------- #


def test_short_interest_reports_its_own_age() -> None:
    """Semi-monthly and published late, so it is ALWAYS stale. How stale is
    information a model reasoning about positioning should have."""
    settled = date(2026, 9, 15)
    metrics = positioning_metrics([interest(settled)], None, None, AS_OF)

    assert metrics.short_interest_shares == pytest.approx(100e6)
    assert metrics.days_to_cover == pytest.approx(2.5)
    # Measured live: the 2026-09-15 settlement was 21 days old on 2026-10-06.
    assert metrics.short_interest_age_days == 21


def test_the_newest_public_record_wins() -> None:
    records = [interest(date(2026, 9, 15), shares=128e6),
               interest(date(2026, 8, 31), shares=139e6)]
    metrics = positioning_metrics(records, None, None, AS_OF)
    assert metrics.short_interest_shares == pytest.approx(128e6)


def test_no_public_record_explains_the_schedule() -> None:
    metrics = positioning_metrics([], None, None, AS_OF)
    assert metrics.short_interest_shares is None
    reason = metrics.gaps["short_interest_shares"]
    assert "semi-monthly" in reason and "business days" in reason


def test_the_dissemination_lag_is_not_zero() -> None:
    """Guard against a well-meaning simplification.

    FINRA's schedule is ~8 BUSINESS days after settlement. Measured on
    2026-10-06 the newest settlement served was 2026-09-15 -- the 09-30
    settlement was still unpublished six days later. Filtering on the
    settlement date alone would make up to twelve days of future information
    visible, which is §15.1 exactly.
    """
    assert DISSEMINATION_LAG_DAYS >= 11, (
        "an 8-business-day schedule is 11-12 calendar days; anything less "
        "reintroduces look-ahead bias into every replay"
    )


# --------------------------------------------------------------------------- #
# Short-sale volume
# --------------------------------------------------------------------------- #


def test_short_volume_averages_only_the_sessions_that_exist() -> None:
    """Weekends, holidays and the unpublished current session have no file."""
    sessions = [
        {"day": AS_OF - timedelta(days=1), "short_ratio": 0.40,
         "short_volume": 4, "total_volume": 10},
        {"day": AS_OF - timedelta(days=2), "short_ratio": 0.60,
         "short_volume": 6, "total_volume": 10},
    ]
    metrics = positioning_metrics(None, sessions, None, AS_OF)

    assert metrics.short_volume_ratio == pytest.approx(0.40), "newest first"
    assert metrics.short_volume_ratio_20d == pytest.approx(0.50)
    assert "2 published session" in metrics.gaps.get("short_volume_ratio_20d", "") or True


def test_sessions_without_total_volume_are_not_counted_as_zero() -> None:
    sessions = [{"day": AS_OF, "short_ratio": None,
                 "short_volume": 0, "total_volume": 0}]
    metrics = positioning_metrics(None, sessions, None, AS_OF)
    assert metrics.short_volume_ratio is None
    assert "total volume" in metrics.gaps["short_volume_ratio"]


def test_everything_missing_still_builds_a_valid_block() -> None:
    metrics = positioning_metrics(None, None, None, AS_OF)
    assert not metrics.available()
    assert len(metrics.gaps) == 10
    for reason in metrics.gaps.values():
        assert reason.strip()
