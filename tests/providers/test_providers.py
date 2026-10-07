"""Cache, point-in-time discipline, and the replay guard.

§15.1 lists look-ahead bias as risk number one: *"Providers that cannot do
point-in-time must RAISE in replay, not degrade quietly."* These are the tests
that make that true rather than intended.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from research_desk.config import Settings
from research_desk.models.market import Bar, BarSeries
from research_desk.models.modes import RunMode
from research_desk.providers.base import (
    NetworkForbiddenError,
    PointInTimeError,
    ProviderError,
    ProviderSpec,
)
from research_desk.providers.cache import Cache, cache_key
from research_desk.providers.ib import IBBarsProvider
from research_desk.providers.limiter import RateLimiter
from research_desk.providers.registry import ProviderRegistry


@pytest.fixture
def cache(tmp_path) -> Cache:
    return Cache(tmp_path / "cache")


def bars_payload(symbol: str, days: int, end: date, *, source: str = "ib") -> dict:
    rows = []
    for i in range(days):
        day = end - timedelta(days=days - 1 - i)
        price = 100.0 + i
        rows.append({
            "day": day.isoformat(), "open": price, "high": price * 1.01,
            "low": price * 0.99, "close": price, "volume": 1e6, "source": source,
        })
    return {"symbol": symbol, "source": source, "bars": rows}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_the_key_is_stable_across_key_order() -> None:
    """§7.4: sha256(provider, method, sorted(kwargs), as_of). Sorted matters --
    an unstable key is a cache that silently never hits."""
    a = cache_key("edgar", "facts", date(2026, 1, 1), {"symbol": "AAPL", "cik": "320193"})
    b = cache_key("edgar", "facts", date(2026, 1, 1), {"cik": "320193", "symbol": "AAPL"})
    assert a == b


def test_the_key_separates_different_as_of_dates() -> None:
    """The point of the whole exercise: 2024's answer is not 2026's."""
    early = cache_key("edgar", "facts", date(2024, 1, 1), {"symbol": "AAPL"})
    late = cache_key("edgar", "facts", date(2026, 1, 1), {"symbol": "AAPL"})
    assert early != late


def test_round_trip(cache: Cache) -> None:
    cache.put("ib", "daily_bars", payload={"hello": "world"}, symbol="SPY")
    entry = cache.get("ib", "daily_bars", symbol="SPY")
    assert entry is not None
    assert entry["payload"] == {"hello": "world"}
    assert entry["fetched_at"]


def test_a_miss_is_none_not_an_exception(cache: Cache) -> None:
    assert cache.get("ib", "daily_bars", symbol="NOPE") is None


def test_writes_leave_no_partial_file(cache: Cache) -> None:
    cache.put("ib", "daily_bars", payload={"x": 1}, symbol="SPY")
    assert not list(cache.root.rglob("*.tmp"))


def test_an_unreadable_entry_is_a_miss_not_a_crash(cache: Cache) -> None:
    """A truncated write must not be mistaken for a bad upstream reply."""
    path = cache.path_for("ib", "daily_bars", symbol="SPY")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated")
    assert cache.get("ib", "daily_bars", symbol="SPY") is None


# --------------------------------------------------------------------------- #
# Point-in-time
# --------------------------------------------------------------------------- #


async def test_bars_are_truncated_at_as_of(cache: Cache) -> None:
    """The guard lives in the provider, not the caller.

    §2's tool-promotion path means a model may eventually request bars at an
    arbitrary date, and that is exactly where look-ahead bias creeps back in.
    """
    end = date(2026, 10, 6)
    cache.put("ib", "daily_bars", payload=bars_payload("SPY", 30, end), symbol="SPY")
    provider = IBBarsProvider(cache)

    cutoff = end - timedelta(days=10)
    series = await provider.daily_bars("SPY", cutoff)

    assert series.last_day is not None and series.last_day <= cutoff
    assert all(b.day <= cutoff for b in series.bars)
    assert len(series) == 20


async def test_a_missing_symbol_names_the_command_that_fixes_it(cache: Cache) -> None:
    """A cache miss here is an operational state, not a bug."""
    provider = IBBarsProvider(cache)
    with pytest.raises(ProviderError, match="make bars"):
        await provider.daily_bars("NOPE", date(2026, 10, 6))


async def test_an_as_of_before_all_history_is_an_error_not_an_empty_series(cache: Cache) -> None:
    """An empty series would compute 38 silent gaps and look like a thin
    symbol rather than a wrong as_of."""
    end = date(2026, 10, 6)
    cache.put("ib", "daily_bars", payload=bars_payload("SPY", 30, end), symbol="SPY")
    provider = IBBarsProvider(cache)

    with pytest.raises(ProviderError, match="none on or before"):
        await provider.daily_bars("SPY", date(2020, 1, 1))


def test_bar_series_rejects_out_of_order_bars() -> None:
    rows = [
        Bar(day=date(2026, 1, 2), open=1, high=1, low=1, close=1, volume=1, source="t"),
        Bar(day=date(2026, 1, 1), open=1, high=1, low=1, close=1, volume=1, source="t"),
    ]
    with pytest.raises(ValueError, match="ascending"):
        BarSeries(symbol="X", source="t", bars=rows)


def test_bar_rejects_incoherent_ohlc() -> None:
    """Catches a mangled parse at the boundary rather than six metrics later --
    which is exactly how a CSV schema change surfaces."""
    with pytest.raises(ValueError, match="high"):
        Bar(day=date(2026, 1, 1), open=10, high=5, low=8, close=9, volume=1, source="t")
    with pytest.raises(ValueError, match="close"):
        Bar(day=date(2026, 1, 1), open=10, high=12, low=8, close=99, volume=1, source="t")


# --------------------------------------------------------------------------- #
# Replay guard
# --------------------------------------------------------------------------- #


def registry_with(mode: RunMode, cache: Cache) -> ProviderRegistry:
    return ProviderRegistry(
        mode=mode, cache=cache, limiter=RateLimiter({}), bars=IBBarsProvider(cache)
    )


async def test_the_cache_reading_bars_provider_works_in_replay(cache: Cache) -> None:
    """It reaches no network, so replay has nothing to forbid. This is the
    reason IB-via-cache is better than a direct fetch, not just a workaround."""
    end = date(2026, 10, 6)
    cache.put("ib", "daily_bars", payload=bars_payload("SPY", 30, end), symbol="SPY")

    registry = registry_with(RunMode.REPLAY, cache)
    series = await registry.daily_bars("SPY", end)
    assert len(series) == 30


async def test_replay_refuses_a_networked_provider(cache: Cache) -> None:
    """§7.4: in replay the cache is the ONLY permitted source."""
    class Networked:
        spec = ProviderSpec(name="finnhub", supports_point_in_time=True,
                            reads_network=True)

        async def daily_bars(self, symbol, as_of):
            raise AssertionError("must never be reached in replay")

    registry = registry_with(RunMode.REPLAY, cache)
    registry._bars = Networked()

    with pytest.raises(NetworkForbiddenError, match="cache only"):
        await registry.daily_bars("SPY", date(2026, 10, 6))


async def test_replay_refuses_a_provider_that_cannot_do_point_in_time(cache: Cache) -> None:
    """RSS is the canonical case: not date-rangeable, so it would hand back
    today's headlines for a replay of 2024."""
    class NotPointInTime:
        spec = ProviderSpec(name="yahoo_rss", supports_point_in_time=False,
                            reads_network=False)

        async def daily_bars(self, symbol, as_of):
            raise AssertionError("must never be reached in replay")

    registry = registry_with(RunMode.REPLAY, cache)
    registry._bars = NotPointInTime()

    with pytest.raises(PointInTimeError, match="supports_point_in_time=False"):
        await registry.daily_bars("SPY", date(2026, 10, 6))


async def test_live_mode_permits_both(cache: Cache) -> None:
    end = date(2026, 10, 6)
    cache.put("ib", "daily_bars", payload=bars_payload("SPY", 5, end), symbol="SPY")
    registry = registry_with(RunMode.LIVE, cache)
    assert len(await registry.daily_bars("SPY", end)) == 5


# --------------------------------------------------------------------------- #
# Limiter
# --------------------------------------------------------------------------- #


async def test_concurrent_callers_serialise_rather_than_burst() -> None:
    """The Stage 4 fan-out case. Without the per-provider lock, four parallel
    analysts each read a stale timestamp and fire together -- which is a burst
    indistinguishable from no rate limiting."""
    limiter = RateLimiter({"edgar": 0.05})
    waits = await asyncio.gather(*[limiter.acquire("edgar") for _ in range(4)])
    assert sum(waits) == pytest.approx(0.15, abs=0.05)


async def test_a_cache_reader_is_never_rate_limited(cache: Cache) -> None:
    """providers.yaml gives ib 1 req/s, written for the real fetch. Applying it
    to file reads would cost 2s per snapshot to throttle nothing."""
    end = date(2026, 10, 6)
    cache.put("ib", "daily_bars", payload=bars_payload("SPY", 5, end), symbol="SPY")
    registry = ProviderRegistry(
        mode=RunMode.LIVE, cache=cache,
        limiter=RateLimiter({"ib": 5.0}), bars=IBBarsProvider(cache),
    )
    import time
    started = time.monotonic()
    for _ in range(3):
        await registry.daily_bars("SPY", end)
    assert time.monotonic() - started < 1.0, "a file read was rate limited"
