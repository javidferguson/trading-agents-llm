"""The Stage 2a exit gate, end to end against a seeded cache.

*"`snapshot SYMBOL=SPY` prints a complete MarketSnapshot with every §7.1 metric
either populated or explicitly None with a reason, no LLM."*

The "no LLM" half is structural -- nothing in this path can reach a model -- so
what is tested here is the "every metric accounted for" half.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research_desk.metrics.snapshot import build_market_snapshot, universe_context
from research_desk.models.modes import RunMode
from research_desk.providers.base import ProviderError
from research_desk.providers.cache import Cache
from research_desk.providers.ib import IBBarsProvider
from research_desk.providers.limiter import RateLimiter
from research_desk.providers.registry import ProviderRegistry

END = date(2026, 10, 6)


def seed(cache: Cache, symbol: str, days: int, *, drift: float = 0.4) -> None:
    """A deterministic series with down days and varying volume.

    Both are load-bearing. A monotonically rising price has no negative
    returns, so downside deviation is correctly unavailable; a constant volume
    has zero dispersion, so the z-score is correctly unavailable. A fixture
    without them tests the gap path by accident rather than the happy path on
    purpose.
    """
    import math
    rows = []
    for i in range(days):
        day = END - timedelta(days=days - 1 - i)
        # Trend plus an oscillation, so roughly a third of sessions are down.
        price = 100.0 + i * drift + 6.0 * math.sin(i / 3.0)
        volume = 2e7 * (1.0 + 0.35 * math.sin(i / 5.0))
        rows.append({
            "day": day.isoformat(), "open": price, "high": price * 1.02,
            "low": price * 0.98, "close": price, "volume": volume, "source": "ib",
        })
    cache.put("ib", "daily_bars",
              payload={"symbol": symbol, "source": "ib", "bars": rows},
              symbol=symbol)


@pytest.fixture
def registry(tmp_path) -> ProviderRegistry:
    cache = Cache(tmp_path / "cache")
    for symbol in ("AAPL", "SPY", "XLK"):
        seed(cache, symbol, 400)
    return ProviderRegistry(mode=RunMode.LIVE, cache=cache,
                            limiter=RateLimiter({}), bars=IBBarsProvider(cache))


async def test_a_full_history_populates_every_metric(registry) -> None:
    snapshot = await build_market_snapshot(
        registry, "AAPL", END, benchmark_symbol="SPY", sector_symbol="XLK"
    )
    populated, missing = snapshot.metric_count()

    assert missing == 0, f"unexpected gaps: {snapshot.all_gaps()}"
    assert populated == 38
    assert snapshot.sources["bars"] == "ib"
    assert snapshot.bars_available == 400


async def test_every_gap_has_a_reason_and_every_reason_a_gap(registry) -> None:
    """The invariant the gate rests on. MetricBlock enforces it at construction,
    so this is a belt-and-braces check at the snapshot level."""
    seed(registry.cache, "TINY", 25)
    snapshot = await build_market_snapshot(
        registry, "TINY", END, benchmark_symbol="SPY", sector_symbol="XLK"
    )

    for key, reason in snapshot.all_gaps().items():
        assert reason and reason.strip(), f"{key} is missing with an empty reason"

    populated, missing = snapshot.metric_count()
    assert populated + missing == 38, "a metric vanished from both sides"


async def test_a_missing_sector_degrades_rather_than_failing(registry) -> None:
    """Supporting series are optional; the symbol's own bars are not."""
    snapshot = await build_market_snapshot(
        registry, "AAPL", END, benchmark_symbol="SPY", sector_symbol="XLU"
    )
    assert snapshot.relative_strength.vs_sector_1m_pct is None
    assert "sector" in snapshot.relative_strength.gaps["vs_sector_1m_pct"]
    # The benchmark still worked, so beta and vs_benchmark survive.
    assert snapshot.risk.beta_vs_spy is not None


async def test_missing_bars_for_the_symbol_itself_raises(registry) -> None:
    """A market report with no prices is not a weaker report, it is a useless
    object -- and the error names the command that fixes it."""
    with pytest.raises(ProviderError, match="make bars"):
        await build_market_snapshot(registry, "NOSUCH", END)


async def test_as_of_is_honoured_end_to_end(registry) -> None:
    """Look-ahead bias is risk number one (§15.1)."""
    cutoff = END - timedelta(days=100)
    snapshot = await build_market_snapshot(registry, "AAPL", cutoff,
                                           benchmark_symbol="SPY")
    assert snapshot.last_bar_day is not None
    assert snapshot.last_bar_day <= cutoff
    assert snapshot.bars_available == 300


async def test_the_snapshot_is_json_serialisable(registry) -> None:
    """It gets rendered into prompts and written to DecisionState JSONL."""
    snapshot = await build_market_snapshot(registry, "AAPL", END,
                                           benchmark_symbol="SPY")
    blob = snapshot.model_dump(mode="json")
    assert blob["symbol"] == "AAPL"
    assert "trend" in blob and "gaps" in blob["trend"]


def test_universe_context_reads_the_shipped_config() -> None:
    from research_desk.config import load_yaml
    universe = load_yaml("universe.yaml")

    benchmark, sector = universe_context(universe, "AAPL")
    assert benchmark == "SPY" and sector == "XLK"

    # The benchmark measured against itself would be a column of zeros.
    benchmark, sector = universe_context(universe, "SPY")
    assert benchmark is None
