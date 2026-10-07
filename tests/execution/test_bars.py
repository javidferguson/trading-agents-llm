"""The IB bars fetcher. Tested without a Gateway, because the bugs are not in IB.

This code lives on the execute side and imports ib_async, so it is exercised
with a fake IB. What is worth testing has nothing to do with the network: it is
contract resolution, the share-class spelling, and the normaliser -- all of
which fail *silently* in production if they are wrong.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from research_desk.execution.bars import (
    CACHE_PROVIDER,
    bar_from_ib,
    fetch_series,
    ib_ticker,
)
from research_desk.providers.cache import Cache
from research_desk.providers.ib import IBBarsProvider


# --------------------------------------------------------------------------- #
# Share-class spellings -- a silently empty series if wrong
# --------------------------------------------------------------------------- #


def test_ib_uses_a_space_for_share_classes() -> None:
    """IB writes BRK.B as 'BRK B'. Everyone else writes it some other way, and
    sending the wrong one does not error -- qualifyContracts just returns
    nothing, which surfaces as an empty price series much later."""
    assert ib_ticker("BRK.B", {"ib": "BRK B", "stooq": "brk-b.us"}) == "BRK B"


def test_an_unmapped_share_class_still_gets_ibs_convention() -> None:
    """So a newly added ticker works before someone writes the mapping."""
    assert ib_ticker("BF.B", None) == "BF B"


def test_ordinary_tickers_are_untouched() -> None:
    assert ib_ticker("AAPL", None) == "AAPL"
    assert ib_ticker("aapl", {}) == "AAPL"


def test_the_shipped_universe_mapping_is_what_ib_wants() -> None:
    from research_desk.config import load_yaml

    for entry in load_yaml("universe.yaml")["symbols"]:
        mapped = (entry.get("provider_symbols") or {}).get("ib")
        if mapped:
            assert "." not in mapped, f"{entry['symbol']}: IB uses a space, not a dot"


# --------------------------------------------------------------------------- #
# Contract resolution
# --------------------------------------------------------------------------- #


class FakeIB:
    def __init__(self, qualified, rows=None):
        self._qualified = qualified
        self._rows = rows or []
        self.requested = None

    async def qualifyContractsAsync(self, *contracts, returnAll=False):
        return self._qualified

    async def reqHistoricalDataAsync(self, contract, **kwargs):
        self.requested = kwargs
        return self._rows


def ib_row(day: str, close: float, volume: float = 1e6):
    return SimpleNamespace(
        date=date.fromisoformat(day), open=close, high=close * 1.01,
        low=close * 0.99, close=close, volume=volume,
    )


async def test_an_unresolvable_symbol_fails_with_the_cause() -> None:
    """qualifyContractsAsync returns a list that may contain None -- an unknown
    symbol does not raise. Unpacking it blindly turned that into an
    AttributeError sixty lines away."""
    ib = FakeIB(qualified=[None])
    with pytest.raises(RuntimeError, match="could not resolve"):
        await fetch_series(ib, "NOSUCH")


async def test_an_empty_qualification_list_is_handled() -> None:
    ib = FakeIB(qualified=[])
    with pytest.raises(RuntimeError, match="could not resolve"):
        await fetch_series(ib, "NOSUCH")


async def test_a_contract_with_no_conid_counts_as_unresolved() -> None:
    ib = FakeIB(qualified=[SimpleNamespace(conId=0, primaryExchange="")])
    with pytest.raises(RuntimeError, match="could not resolve"):
        await fetch_series(ib, "WEIRD")


async def test_no_bars_is_distinguished_from_no_contract() -> None:
    """Different causes, different fixes: one is a ticker spelling, the other
    is a data permission."""
    ib = FakeIB(qualified=[SimpleNamespace(conId=265598, primaryExchange="NASDAQ")], rows=[])
    with pytest.raises(RuntimeError, match="permission"):
        await fetch_series(ib, "AAPL")


async def test_end_date_time_is_timezone_aware() -> None:
    """§14, measured on the ORB engine: a naive endDateTime is resolved by IB
    in a zone of its own choosing. For daily bars the error is a whole session.
    """
    ib = FakeIB(
        qualified=[SimpleNamespace(conId=1, primaryExchange="ARCA")],
        rows=[ib_row("2026-10-05", 100.0)],
    )
    await fetch_series(ib, "SPY", end=date(2026, 10, 6))

    sent = ib.requested["endDateTime"]
    assert isinstance(sent, datetime)
    assert sent.tzinfo is not None, "a naive endDateTime is the measured bug"
    assert sent.tzinfo == ZoneInfo("America/New_York")


async def test_the_cache_keeps_the_canonical_ticker_not_ibs() -> None:
    """providers/ib.py looks the cache up by the universe's symbol. Storing
    'BRK B' there would be a permanent, silent miss."""
    ib = FakeIB(
        qualified=[SimpleNamespace(conId=1, primaryExchange="NYSE")],
        rows=[ib_row("2026-10-05", 405.0)],
    )
    series = await fetch_series(ib, "BRK.B", provider_symbols={"ib": "BRK B"})
    assert series.symbol == "BRK.B"


# --------------------------------------------------------------------------- #
# The normaliser -- the single conversion point
# --------------------------------------------------------------------------- #


def test_bar_from_ib_records_its_source() -> None:
    """§7.4: when the primary fails over the numbers differ slightly, and you
    want to know which a decision was made on."""
    bar = bar_from_ib(ib_row("2026-10-05", 100.0))
    assert bar.source == CACHE_PROVIDER
    assert bar.day == date(2026, 10, 5)


def test_ib_sentinel_volume_becomes_zero_not_negative() -> None:
    """IB sends -1 for 'no volume data'. A negative volume would sail into
    dollar ADV and produce a negative liquidity figure."""
    bar = bar_from_ib(ib_row("2026-10-05", 100.0, volume=-1))
    assert bar.volume == 0.0


def test_a_datetime_date_is_narrowed_to_a_day() -> None:
    row = ib_row("2026-10-05", 100.0)
    row.date = datetime(2026, 10, 5, 16, 0)
    assert bar_from_ib(row).day == date(2026, 10, 5)


@pytest.mark.parametrize("raw", ["2026-10-05", "20261005", "2026-10-05 16:00:00"])
def test_both_of_ibs_date_spellings_parse(raw: str) -> None:
    """formatDate=1 gives ISO, but IB is not consistent across endpoints and
    Python 3.11+ accepts the compact form too. Accept both rather than
    depending on which one today's TWS build happens to send."""
    row = ib_row("2026-10-05", 100.0)
    row.date = raw
    assert bar_from_ib(row).day == date(2026, 10, 5)


def test_an_unparseable_date_fails_loudly() -> None:
    """Silently dropping a bar would leave a hole no metric could see."""
    row = ib_row("2026-10-05", 100.0)
    row.date = "not-a-date"
    with pytest.raises(ValueError):
        bar_from_ib(row)


async def test_fetched_bars_round_trip_through_the_cache(tmp_path) -> None:
    """The bridge end to end: what the execute side writes, decide can read."""
    from research_desk.execution.bars import CACHE_METHOD

    ib = FakeIB(
        qualified=[SimpleNamespace(conId=1, primaryExchange="ARCA")],
        rows=[ib_row(f"2026-09-{d:02d}", 100.0 + d) for d in range(1, 11)],
    )
    series = await fetch_series(ib, "SPY")

    cache = Cache(tmp_path)
    cache.put(CACHE_PROVIDER, CACHE_METHOD,
              payload={"symbol": series.symbol, "source": series.source,
                       "bars": [b.model_dump(mode="json") for b in series.bars]},
              as_of=None, symbol=series.symbol)

    read_back = await IBBarsProvider(cache).daily_bars("SPY", date(2026, 9, 10))
    assert len(read_back) == 10
    assert read_back.source == CACHE_PROVIDER
    assert read_back.bars[-1].close == pytest.approx(110.0)
