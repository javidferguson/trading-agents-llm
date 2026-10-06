#!/usr/bin/env python3
"""Write SYNTHETIC daily bars into the provider cache, for development only.

!!  THIS IS NOT MARKET DATA.  !!

Every bar it writes carries ``source="fixture"``, and `desk snapshot` prints a
warning whenever a snapshot is built from them. That is deliberate: synthetic
prices that are mistaken for real ones would produce a Stage 8 equity curve
that is not merely wrong but fictional, which §7.5 already warns about for
survivorship bias and is doubly true here.

Why it exists: the real bars come from IB via ``make bars``, which needs the
Gateway up and credentials set. Until then this lets the Stage 2 metric
pipeline be built, demonstrated and tested deterministically. Replace it with
real bars the moment the Gateway is available -- `make bars` overwrites these.

    python scripts/seed_fixture_bars.py
    python scripts/seed_fixture_bars.py --symbols AAPL --days 300
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_desk.config import load_settings, load_yaml  # noqa: E402
from research_desk.providers.cache import Cache  # noqa: E402

FIXTURE_SOURCE = "fixture"

#: Per-symbol (start price, annual drift, annual vol). Chosen so the resulting
#: metrics land in plausible ranges -- an RSI of 71 and a 0.94 range percentile
#: exercise the prompts more usefully than a flat line would.
PROFILES = {
    "SPY":  (520.0, 0.11, 0.14),
    "AAPL": (195.0, 0.14, 0.24),
    "MSFT": (390.0, 0.16, 0.23),
    "NVDA": (780.0, 0.38, 0.48),
    "AVGO": (1250.0, 0.30, 0.38),
    "AMZN": (175.0, 0.18, 0.30),
    "GOOGL": (150.0, 0.13, 0.27),
    "META": (480.0, 0.22, 0.34),
    "JPM":  (195.0, 0.10, 0.21),
    "XOM":  (112.0, 0.05, 0.23),
    "UNH":  (505.0, 0.04, 0.25),
    "JNJ":  (158.0, 0.03, 0.16),
    "MRVL": (72.0, 0.25, 0.45),
    "TSM":  (135.0, 0.28, 0.33),
    "BRK.B": (405.0, 0.09, 0.15),
    "XLK":  (215.0, 0.18, 0.21),
    "XLY":  (185.0, 0.12, 0.22),
    "XLC":  (82.0, 0.15, 0.22),
    "XLF":  (42.0, 0.09, 0.19),
    "XLE":  (92.0, 0.06, 0.26),
    "XLV":  (145.0, 0.05, 0.16),
}
DEFAULT = (100.0, 0.08, 0.25)


def trading_days(end: date, count: int) -> list[date]:
    """Weekdays only. Holidays are not modelled; the metrics do not care."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def synth(symbol: str, days: int, end: date) -> list[dict]:
    start_price, drift, vol = PROFILES.get(symbol.upper(), DEFAULT)
    # Seeded per symbol so repeated runs and test fixtures are identical.
    rng = random.Random(f"research-desk/{symbol.upper()}")

    daily_drift = drift / 252.0
    daily_vol = vol / math.sqrt(252.0)

    price = start_price
    out = []
    for day in trading_days(end, days):
        shock = rng.gauss(daily_drift, daily_vol)
        close = max(price * (1.0 + shock), 0.01)
        intraday = abs(rng.gauss(0, daily_vol * 0.6))
        high = max(price, close) * (1.0 + intraday)
        low = min(price, close) * (1.0 - intraday)
        out.append({
            "day": day.isoformat(),
            "open": round(price, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close, 4),
            "volume": float(int(rng.uniform(4e6, 9e7))),
            "source": FIXTURE_SOURCE,
        })
        price = close
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--days", type=int, default=1300, help="~5 years of weekdays")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    universe = load_yaml("universe.yaml")
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = sorted(
            {e["symbol"] for e in universe["symbols"]}
            | {universe["benchmark"]}
            | {e["sector_etf"] for e in universe["symbols"] if e.get("sector_etf")}
        )

    end = date.fromisoformat(args.end) if args.end else date.today()
    cache = Cache(load_settings().cache_dir)

    print("!! SYNTHETIC DATA -- not market prices. source=fixture on every bar.\n")
    for symbol in symbols:
        bars = synth(symbol, args.days, end)
        cache.put(
            "ib", "daily_bars",
            payload={
                "symbol": symbol,
                "source": FIXTURE_SOURCE,
                "adjusted": False,
                "synthetic": True,
                "bars": bars,
            },
            as_of=None,
            symbol=symbol,
        )
        print(f"  {symbol:<8} {len(bars):>5} bars  {bars[0]['day']} .. {bars[-1]['day']}")

    print(f"\nWrote {len(symbols)} series to {cache.root / 'ib'}")
    print("Replace with real data when the Gateway is up:  make bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
