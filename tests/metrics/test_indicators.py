"""§7.1 arithmetic, against hand-computed values and stated invariants.

These matter more than most tests here, because this is the module §2 exists to
protect: *"Never let an LLM do arithmetic."* The whole design routes every
number through here, so a wrong RSI is not a wrong metric -- it is a wrong
premise handed to fourteen model calls that will each reason confidently from
it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research_desk.metrics import indicators as ind
from research_desk.models.market import Bar, BarSeries, MeanReversionMetrics, TrendMetrics


def series_from(closes: list[float], *, volume: float = 1_000_000.0) -> BarSeries:
    """A BarSeries with coherent OHLC around each close."""
    bars = []
    day = date(2024, 1, 1)
    for close in closes:
        bars.append(Bar(
            day=day, open=close, high=close * 1.01, low=close * 0.99,
            close=close, volume=volume, source="test",
        ))
        day += timedelta(days=1)
    return BarSeries(symbol="TEST", source="test", bars=bars)


# --------------------------------------------------------------------------- #
# Primitives, hand-checked
# --------------------------------------------------------------------------- #


def test_sma_is_the_mean_of_the_window() -> None:
    assert ind.sma([1, 2, 3, 4, 5], 5) == 3.0
    assert ind.sma([1, 2, 3, 4, 5], 2) == 4.5


def test_sma_refuses_a_window_longer_than_the_data() -> None:
    """Not a partial average. A 200-day SMA over 40 bars is a different
    statistic wearing the same name."""
    assert ind.sma([1, 2, 3], 10) is None


def test_ema_is_seeded_with_the_sma_not_the_first_point() -> None:
    """Seeding from one bar makes a hundred sessions of MACD depend on it.

    Hand-computed for window=2 over [1, 2, 3]:
      seed  = (1 + 2) / 2 = 1.5
      alpha = 2 / (2 + 1) = 2/3
      next  = 2/3 * 3 + 1/3 * 1.5 = 2.0 + 0.5 = 2.5
    """
    assert ind.ema_series([1, 2, 3], 2) == pytest.approx([1.5, 2.5])
    assert ind.ema([1, 2, 3], 2) == pytest.approx(2.5)


def test_returns_are_period_over_period() -> None:
    assert ind.returns([100, 110, 99]) == pytest.approx([0.1, -0.1])


def test_total_return_uses_the_right_endpoints() -> None:
    # lookback=2 over [100, x, 121] compares 121 against 100.
    assert ind.total_return_pct([100, 150, 121], 2) == pytest.approx(21.0)


def test_rsi_is_100_when_every_day_gains() -> None:
    """The defined limit, not a sentinel -- it means no down days at all."""
    assert ind.rsi(list(range(1, 40)), 14) == 100.0


def test_rsi_is_0_when_every_day_loses() -> None:
    assert ind.rsi(list(range(40, 1, -1)), 14) == pytest.approx(0.0)


def test_rsi_sits_midrange_on_an_alternating_series() -> None:
    closes = [100 + (1 if i % 2 else -1) for i in range(60)]
    value = ind.rsi(closes, 14)
    assert value is not None and 30 < value < 70


def test_wilder_average_is_not_a_simple_mean() -> None:
    """Using an SMA here gives numbers close enough to look right and different
    enough to disagree with every charting package."""
    # A flat seed window cannot distinguish the two -- with [1.0]*14 + [10.0]
    # both give 23/14 exactly. Needs a varied series to tell them apart.
    values = [float(i) for i in range(1, 15)] + [100.0]

    simple = sum(values[-14:]) / 14                      # (2..14 + 100) / 14
    wilder = ind.wilder_average(values, 14)

    # Wilder: seed = mean(1..14) = 7.5, then (7.5 * 13 + 100) / 14
    assert wilder == pytest.approx((7.5 * 13 + 100.0) / 14)
    assert simple == pytest.approx((sum(range(2, 15)) + 100.0) / 14)
    assert wilder != pytest.approx(simple), (
        "if these ever agree, the smoothing has been replaced by an SMA and "
        "every ATR and RSI will quietly disagree with every charting package"
    )


def test_max_drawdown_finds_the_worst_peak_to_trough() -> None:
    assert ind.max_drawdown_pct([100, 120, 60, 90]) == pytest.approx(-50.0)
    assert ind.max_drawdown_pct([100, 101, 102]) == pytest.approx(0.0)


def test_percentile_rank_locates_a_value_in_its_own_history() -> None:
    """§7.1: percentiles, never raw scores -- '-1.7' means nothing to an 8B model."""
    assert ind.percentile_rank([1, 2, 3, 4], 0) == 0.0
    assert ind.percentile_rank([1, 2, 3, 4], 5) == 1.0
    assert ind.percentile_rank([1, 2, 3, 4], 3) == 0.5


def test_beta_against_itself_is_one() -> None:
    rets = [0.01, -0.02, 0.015, 0.0, -0.005] * 10
    assert ind.beta(rets, rets) == pytest.approx(1.0)


def test_beta_needs_enough_overlap_to_mean_anything() -> None:
    assert ind.beta([0.01] * 5, [0.01] * 5) is None


def test_downside_deviation_ignores_up_days() -> None:
    """Standard deviation punishes upside surprise like a drawdown, which is
    not how anyone experiences a portfolio."""
    mixed = [0.05, -0.01, 0.06, -0.02, 0.07, -0.015]
    only_down = [-0.01, -0.02, -0.015]
    assert ind.downside_deviation_pct(mixed) == pytest.approx(
        ind.downside_deviation_pct(only_down)
    )


# --------------------------------------------------------------------------- #
# The gap contract -- the Stage 2 exit gate
# --------------------------------------------------------------------------- #


def test_a_short_series_explains_every_missing_metric() -> None:
    metrics = ind.trend_metrics(series_from([100.0 + i for i in range(30)]))

    assert metrics.sma_20 is not None, "20 bars of 30 is computable"
    assert metrics.sma_200 is None
    assert "needs 200 daily bars, have 30" in metrics.gaps["sma_200"]
    # And nothing is None without a reason -- MetricBlock would have refused to
    # construct, so reaching this line at all is part of the assertion.
    assert set(metrics.gaps) >= {"sma_50", "sma_200", "cross_state", "pct_52w_range"}


def test_cascading_gaps_name_their_dependency() -> None:
    """'needs sma_200, which is unavailable' beats repeating the bar count:
    it tells you which single fix unblocks several metrics."""
    metrics = ind.trend_metrics(series_from([100.0] * 30))
    assert metrics.gaps["price_vs_sma_200_pct"] == "needs sma_200, which is unavailable"


def test_a_one_bar_series_still_produces_a_valid_block() -> None:
    """Degrade, never raise. A snapshot with one bar is useless but honest."""
    metrics = ind.trend_metrics(series_from([100.0]))
    assert metrics.close == 100.0
    assert len(metrics.gaps) >= 10


def test_metrics_labelled_1y_actually_require_a_year() -> None:
    """The error this guards: computing a drawdown over 30 bars and calling it
    a 1-year drawdown."""
    short = ind.risk_metrics(series_from([100.0 + i for i in range(100)]))
    assert short.downside_deviation_pct is None
    assert "252" in short.gaps["downside_deviation_pct"]

    trend = ind.trend_metrics(series_from([100.0 + i for i in range(100)]))
    assert trend.max_drawdown_1y_pct is None
    assert "252" in trend.gaps["max_drawdown_1y_pct"]


def test_absent_volume_is_reported_not_treated_as_zero() -> None:
    """IB sends -1 for 'no volume data'. A quiet zero would corrupt dollar ADV
    and the volume z-score while looking like a measurement (§7.5)."""
    metrics = ind.risk_metrics(series_from([100.0 + i for i in range(60)], volume=0.0))

    for name in ("dollar_adv_20d", "volume_zscore", "amihud_illiquidity"):
        assert getattr(metrics, name) is None
        assert "no volume data" in metrics.gaps[name]


def test_a_flat_series_does_not_divide_by_zero() -> None:
    """Degenerate but real: a halted or brand-new listing."""
    flat = ind.mean_reversion_metrics(series_from([100.0] * 60))
    assert flat.bollinger_pct_b is None
    assert flat.gaps["bollinger_pct_b"]

    trend = ind.trend_metrics(series_from([100.0] * 300))
    assert trend.pct_52w_range is None


def test_relative_strength_without_a_sector_says_so() -> None:
    base = series_from([100.0 + i for i in range(300)])
    metrics = ind.relative_strength_metrics(base, base, None, benchmark_symbol="SPY")

    assert metrics.vs_benchmark_1m_pct == pytest.approx(0.0), "itself vs itself"
    assert metrics.vs_sector_1m_pct is None
    assert metrics.gaps["vs_sector_1m_pct"] == "sector ETF series unavailable"
    assert metrics.sector_symbol is None


def test_amihud_is_scaled_to_be_legible() -> None:
    """The textbook form gives ~6e-7 for a megacap, which prints as 0.0000 and
    tells a model nothing."""
    metrics = ind.risk_metrics(series_from([100.0 + i * 0.5 for i in range(60)],
                                           volume=50_000_000.0))
    assert metrics.amihud_illiquidity is not None
    assert 1e-4 < metrics.amihud_illiquidity < 1e4, "must be humanly readable"
