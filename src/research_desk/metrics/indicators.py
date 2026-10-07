"""Every number. **No model ever computes one of these, and no prompt contains a formula.**

> *"An 8B model will confidently produce a wrong RSI. It is genuinely good at
> 'RSI is 71 and price is at the upper Bollinger band -- that's stretched.'
> This single rule is the difference between local models being usable and
> useless here."* -- architecture §2

Two conventions hold throughout, and both exist to serve the Stage 2 exit gate
(*every metric populated or explicitly ``None`` with a reason*):

* **A metric with insufficient history is ``None`` with the shortfall stated**,
  never a value computed from whatever happened to be available. A 200-day SMA
  over 40 bars is not a worse estimate, it is a different statistic wearing the
  same name.
* **Nothing is silently zero.** §7.5 is emphatic: a zero is indistinguishable
  from a real measurement and corrupts every ablation downstream.

Stdlib only. ``numpy`` arrives at Stage 9 for the reflection dot products; a
few thousand daily bars do not need it, and the formulas read better without
vectorisation.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from ..models.market import (
    BarSeries,
    MeanReversionMetrics,
    RelativeStrengthMetrics,
    RiskMetrics,
    TrendMetrics,
)

logger = logging.getLogger(__name__)

#: Trading-day conventions. Calendar months are the wrong unit for a bar series.
MONTH = 21
QUARTER = 63
YEAR = 252

#: Annualisation factor for daily volatility.
ANNUALISE = math.sqrt(YEAR)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def sma(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema_series(values: Sequence[float], window: int) -> list[float] | None:
    """Standard EMA, seeded with the SMA of the first ``window`` points.

    Seeding matters: starting from the first value alone makes early output
    depend heavily on one bar, which then propagates into MACD for a hundred
    sessions.
    """
    if len(values) < window:
        return None
    alpha = 2.0 / (window + 1)
    out = [sum(values[:window]) / window]
    for value in values[window:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def ema(values: Sequence[float], window: int) -> float | None:
    series = ema_series(values, window)
    return series[-1] if series else None


def returns(values: Sequence[float]) -> list[float]:
    """Simple period returns. Undefined where the prior close is non-positive."""
    out = []
    for before, after in zip(values, values[1:]):
        out.append((after / before) - 1.0 if before > 0 else 0.0)
    return out


def total_return_pct(values: Sequence[float], lookback: int) -> float | None:
    if len(values) < lookback + 1:
        return None
    start, end = values[-(lookback + 1)], values[-1]
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def annualised_vol_pct(daily_returns: Sequence[float]) -> float | None:
    if len(daily_returns) < 2:
        return None
    return statistics.stdev(daily_returns) * ANNUALISE * 100.0


def true_ranges(series: BarSeries) -> list[float]:
    """Wilder's true range: the widest of today's range and the two gaps."""
    out = []
    for previous, current in zip(series.bars, series.bars[1:]):
        out.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return out


def wilder_average(values: Sequence[float], window: int) -> float | None:
    """Wilder's smoothing -- what ATR and RSI actually use.

    Not a simple moving average. Using one gives numbers close enough to look
    right and different enough to disagree with every charting package, which
    is the worst possible outcome for a metric a human is going to sanity-check.
    """
    if len(values) < window:
        return None
    average = sum(values[:window]) / window
    for value in values[window:]:
        average = (average * (window - 1) + value) / window
    return average


def rsi(closes: Sequence[float], window: int = 14) -> float | None:
    if len(closes) < window + 1:
        return None
    changes = [after - before for before, after in zip(closes, closes[1:])]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    avg_gain = wilder_average(gains, window)
    avg_loss = wilder_average(losses, window)
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        # Uninterrupted gains over the window. 100 is the defined limit, not a
        # sentinel -- but it is worth knowing it means "no down days at all".
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def max_drawdown_pct(values: Sequence[float]) -> float | None:
    """Worst peak-to-trough decline, as a negative percentage."""
    if len(values) < 2:
        return None
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value / peak - 1.0))
    return worst * 100.0


def percentile_rank(values: Sequence[float], target: float) -> float | None:
    """Where ``target`` sits in ``values``, 0..1.

    §7.1's rule: *"Feed percentiles, never raw tone, and never an adjective."*
    A tone of -1.7 means nothing to an 8B model; "12th percentile of its own
    last year" means something.
    """
    if not values:
        return None
    below = sum(1 for v in values if v < target)
    return below / len(values)


def beta(asset_returns: Sequence[float], market_returns: Sequence[float]) -> float | None:
    paired = list(zip(asset_returns, market_returns))
    if len(paired) < 30:
        return None
    asset = [a for a, _ in paired]
    market = [m for _, m in paired]
    market_var = statistics.variance(market)
    if market_var == 0:
        return None
    covariance = statistics.covariance(asset, market)
    return covariance / market_var


def downside_deviation_pct(daily_returns: Sequence[float]) -> float | None:
    """Annualised dispersion of losing days only.

    Standard deviation punishes upside surprise identically to a drawdown,
    which is not how anyone experiences a portfolio.
    """
    negatives = [r for r in daily_returns if r < 0]
    if len(negatives) < 2:
        return None
    return statistics.stdev(negatives) * ANNUALISE * 100.0


# --------------------------------------------------------------------------- #
# Gap-tracking builder
# --------------------------------------------------------------------------- #


class _Block:
    """Accumulates metric values and the reason for every one that is missing.

    This exists so the Stage 2 gate is satisfied by construction rather than by
    remembering. ``compute`` either stores a number or records why there isn't
    one; there is no third path, and ``MetricBlock`` refuses to build if a field
    ends up ``None`` with nothing in ``gaps``.
    """

    def __init__(self, bars: int):
        self.bars = bars
        self.values: dict[str, Any] = {}
        self.gaps: dict[str, str] = {}

    def compute(
        self,
        name: str,
        needs: int,
        fn: Callable[[], Any],
        *,
        note: str | None = None,
    ) -> Any:
        if self.bars < needs:
            self.gaps[name] = (
                note or f"needs {needs} daily bars, have {self.bars}"
            )
            return None
        try:
            value = fn()
        except (ZeroDivisionError, statistics.StatisticsError, ValueError) as exc:
            self.gaps[name] = f"not computable: {type(exc).__name__}: {exc}"
            return None
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            self.gaps[name] = note or "not computable from the available history"
            return None
        self.values[name] = value
        return value

    def unavailable(self, name: str, reason: str) -> None:
        self.gaps[name] = reason

    def build(self, model: type) -> Any:
        return model(**self.values, gaps=self.gaps)


# --------------------------------------------------------------------------- #
# §7.1 metric blocks
# --------------------------------------------------------------------------- #


def trend_metrics(series: BarSeries) -> TrendMetrics:
    """Price and trend. The 50/200 cross state is the B3 baseline's input."""
    closes = series.closes
    block = _Block(len(series))

    close = block.compute("close", 1, lambda: closes[-1])

    for window in (20, 50, 200):
        value = block.compute(f"sma_{window}", window, lambda w=window: sma(closes, w))
        if value is not None and close:
            block.compute(
                f"price_vs_sma_{window}_pct",
                window,
                lambda v=value: (close / v - 1.0) * 100.0,
            )
        else:
            block.unavailable(
                f"price_vs_sma_{window}_pct", f"needs sma_{window}, which is unavailable"
            )

    def _cross() -> float | None:
        fast, slow = sma(closes, 50), sma(closes, 200)
        if fast is None or slow is None:
            return None
        return 1.0 if fast > slow else -1.0
    block.compute("cross_state", 200, _cross)

    # 12-1 momentum: the academically standard definition. The excluded month
    # matters -- recent-month reversal contaminates it, and keeping both this
    # and reversal_1m separate stops a model fusing them into one story.
    def _momentum_12_1() -> float | None:
        if len(closes) < YEAR + 1:
            return None
        start, end = closes[-(YEAR + 1)], closes[-(MONTH + 1)]
        return (end / start - 1.0) * 100.0 if start > 0 else None
    block.compute("momentum_12_1_pct", YEAR + 1, _momentum_12_1)

    block.compute("reversal_1m_pct", MONTH + 1, lambda: total_return_pct(closes, MONTH))

    def _range_position() -> float | None:
        window = closes[-YEAR:]
        low, high = min(window), max(window)
        if high == low:
            return None
        return (closes[-1] - low) / (high - low)
    block.compute("pct_52w_range", YEAR, _range_position)

    # YEAR, not MONTH: computing this over 30 bars and calling it a 1-year
    # drawdown is the exact error the module docstring warns about -- a
    # different statistic wearing the same name. The label sets the requirement.
    block.compute(
        "max_drawdown_1y_pct", YEAR, lambda: max_drawdown_pct(closes[-YEAR:])
    )
    return block.build(TrendMetrics)


def risk_metrics(series: BarSeries, benchmark: BarSeries | None = None) -> RiskMetrics:
    """Risk and liquidity. ATR feeds stop distance, so it feeds position sizing (§9)."""
    closes = series.closes
    daily = returns(closes)
    block = _Block(len(series))

    atr = block.compute("atr_14", 15, lambda: wilder_average(true_ranges(series), 14))
    if atr is not None and closes[-1] > 0:
        block.compute("atr_14_pct_of_price", 15, lambda: atr / closes[-1] * 100.0)
    else:
        block.unavailable("atr_14_pct_of_price", "needs atr_14, which is unavailable")

    vol20 = block.compute("realized_vol_20d_pct", 21, lambda: annualised_vol_pct(daily[-20:]))
    vol60 = block.compute("realized_vol_60d_pct", 61, lambda: annualised_vol_pct(daily[-60:]))
    if vol20 is not None and vol60:
        block.compute("vol_ratio_20_60", 61, lambda: vol20 / vol60)
    else:
        block.unavailable("vol_ratio_20_60", "needs both 20d and 60d realized vol")

    # Slices a year, so it requires a year -- same reasoning as the drawdown.
    block.compute(
        "downside_deviation_pct", YEAR, lambda: downside_deviation_pct(daily[-YEAR:])
    )

    if benchmark is None:
        block.unavailable("beta_vs_spy", "benchmark series unavailable")
    else:
        def _beta() -> float | None:
            # Align on dates: a missing session in either series would
            # otherwise silently offset the pairing and produce a plausible,
            # wrong beta.
            common = {b.day: b.close for b in benchmark.bars}
            pairs = [(b.day, b.close) for b in series.bars if b.day in common]
            if len(pairs) < 31:
                return None
            asset = returns([c for _, c in pairs])
            market = returns([common[d] for d, _ in pairs])
            return beta(asset[-YEAR:], market[-YEAR:])
        # No horizon in the name, so a shorter history is honest rather than
        # mislabelled: it uses up to a year of overlapping sessions and floors
        # at 30, below which the estimate is noise.
        block.compute("beta_vs_spy", 31, _beta,
                      note="needs 30+ overlapping sessions with the benchmark")

    volumes = [b.volume for b in series.bars]
    dollar_volumes = [b.close * b.volume for b in series.bars]

    if any(v > 0 for v in volumes[-20:]):
        block.compute("dollar_adv_20d", 20, lambda: sum(dollar_volumes[-20:]) / 20)

        def _volume_z() -> float | None:
            window = volumes[-20:]
            spread = statistics.stdev(window)
            if spread == 0:
                return None
            return (volumes[-1] - statistics.mean(window)) / spread
        block.compute("volume_zscore", 21, _volume_z)

        def _amihud() -> float | None:
            # Amihud illiquidity, scaled to be READABLE: basis points of
            # absolute return per $1bn of dollar volume.
            #
            # The textbook form (|return| per $1m) gives ~6e-7 for a megacap,
            # which formats as "0.0000" in a prompt and tells a model nothing.
            # §7.1's rule about feeding percentiles rather than raw scores is
            # the same problem: the number has to be legible to be useful.
            # Higher means thinner -- more price impact per dollar traded.
            samples = [
                (abs(r) * 10_000.0) / (dv / 1e9)
                for r, dv in zip(daily[-20:], dollar_volumes[-20:])
                if dv > 0
            ]
            return statistics.mean(samples) if samples else None
        block.compute("amihud_illiquidity", 21, _amihud)
    else:
        for name in ("dollar_adv_20d", "volume_zscore", "amihud_illiquidity"):
            # IB reports -1 for "no volume data", which bar_from_ib stores as 0.
            # That is absence, not a quiet zero -- say so.
            block.unavailable(name, "no volume data in the last 20 bars")

    return block.build(RiskMetrics)


def mean_reversion_metrics(series: BarSeries) -> MeanReversionMetrics:
    """Mean reversion. Deliberately several overlapping measures, not one."""
    closes = series.closes
    block = _Block(len(series))

    block.compute("rsi_14", 15, lambda: rsi(closes, 14))
    fast = block.compute("ema_12", 12, lambda: ema(closes, 12))
    slow = block.compute("ema_26", 26, lambda: ema(closes, 26))

    if fast is not None and slow is not None:
        macd_value = block.compute("macd", 26, lambda: fast - slow)

        def _signal() -> float | None:
            fast_series = ema_series(closes, 12)
            slow_series = ema_series(closes, 26)
            if not fast_series or not slow_series:
                return None
            # Align the two EMAs: ema_series drops the first `window-1` points,
            # so the 12 and 26 series start at different offsets.
            offset = len(fast_series) - len(slow_series)
            line = [f - s for f, s in zip(fast_series[offset:], slow_series)]
            return ema(line, 9)
        signal = block.compute("macd_signal", 35, _signal)

        if macd_value is not None and signal is not None:
            block.compute("macd_histogram", 35, lambda: macd_value - signal)
        else:
            block.unavailable("macd_histogram", "needs both macd and its signal line")
    else:
        for name in ("macd", "macd_signal", "macd_histogram"):
            block.unavailable(name, "needs both the 12- and 26-period EMA")

    def _pct_b() -> float | None:
        window = closes[-20:]
        middle = statistics.mean(window)
        spread = statistics.stdev(window)
        if spread == 0:
            return None
        lower, upper = middle - 2 * spread, middle + 2 * spread
        return (closes[-1] - lower) / (upper - lower)
    block.compute("bollinger_pct_b", 20, _pct_b)

    def _vwap_distance() -> float | None:
        # A true VWAP needs intraday prints. This is the standard daily
        # approximation -- volume-weighted close over 20 sessions -- and it is
        # labelled as a 20-day measure rather than "the" VWAP for that reason.
        window = series.bars[-20:]
        traded = sum(b.volume for b in window)
        if traded <= 0:
            return None
        vwap = sum(b.close * b.volume for b in window) / traded
        return (closes[-1] / vwap - 1.0) * 100.0 if vwap > 0 else None
    block.compute("dist_from_vwap_20d_pct", 20, _vwap_distance,
                  note="needs 20 bars with volume data")

    return block.build(MeanReversionMetrics)


def relative_strength_metrics(
    series: BarSeries,
    benchmark: BarSeries | None,
    sector: BarSeries | None,
    *,
    benchmark_symbol: str | None = None,
    sector_symbol: str | None = None,
) -> RelativeStrengthMetrics:
    """Return minus the benchmark's, and minus the sector's, over three horizons.

    *"The largest gap in the earlier draft. Absolute return tells you almost
    nothing in a trending market."* (§7.1)
    """
    block = _Block(len(series))
    horizons = (("1m", MONTH), ("3m", QUARTER), ("12m", YEAR))

    for label, other, missing_reason in (
        ("benchmark", benchmark, "benchmark series unavailable"),
        ("sector", sector, "sector ETF series unavailable"),
    ):
        for suffix, lookback in horizons:
            name = f"vs_{label}_{suffix}_pct"
            if other is None:
                block.unavailable(name, missing_reason)
                continue

            def _excess(lb=lookback, o=other) -> float | None:
                mine = total_return_pct(series.closes, lb)
                theirs = total_return_pct(o.closes, lb)
                if mine is None or theirs is None:
                    return None
                return mine - theirs
            block.compute(
                name, lookback + 1, _excess,
                note=f"needs {lookback + 1} bars in both series",
            )

    values = dict(block.values)
    values["benchmark_symbol"] = benchmark_symbol
    values["sector_symbol"] = sector_symbol
    return RelativeStrengthMetrics(**values, gaps=block.gaps)
