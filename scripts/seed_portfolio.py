"""Generate or re-mark ``config/portfolio.yaml`` from the bars cache. No IB.

Two jobs, one file:

* ``--seed`` builds the Stage 6 gate's book from scratch. The share counts come
  from ``SEED_BOOK`` below; every *price* comes from the cached daily bars, so
  the weights are arithmetic over real closes rather than numbers chosen to
  look tidy.
* ``--refresh`` re-marks the existing book's ``last_price`` against the newest
  cached bar and moves ``as_of`` to the OLDEST mark it used. Oldest, not today:
  re-marking against a stale cache must not reset the staleness clock, which is
  the whole point of the check in ``PortfolioSnapshot.is_stale``.

Reads the cache directly and never constructs a provider, so it cannot fetch,
cannot hit IB, and runs with the Gateway down.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_desk.models.portfolio import PortfolioSnapshot, Position  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache" / "ib"
OUT_PATH = REPO_ROOT / "config" / "portfolio.yaml"

#: Cash, chosen so equity lands on a round $250,000 with the seeded shares at
#: the closes below. Recomputed exactly in ``seed()``; this is only the target.
TARGET_EQUITY = 250_000.0

#: How far back the average cost is taken from, in cached sessions. ~6 months,
#: so the book shows real unrealised P&L rather than a flat cost basis.
COST_BASIS_LOOKBACK_BARS = 120

#: The seeded book, designed so the drift table and every compliance rule have
#: something real to bite on. Deliberately *not* a tidy book.
#:
#: Regrown from 8 to 15 positions on 2026-10-07 when max_positions went 8 -> 15
#: alongside the universe widening. The properties below are the point; the
#: share counts are just whatever prices them at those weights on the day.
#:
#:   * NVDA and MSFT sit AT max_position_pct (10%), so an add to either is
#:     blocked by the position cap and nothing else. MSFT is 4x its 2.5% target,
#:     NVDA 2x its 5.00% one.
#:   * TSM is well UNDER its 5.00% target at ~2%, so there is one symbol where a
#:     BUY actually sizes. Without it the gate never exercises sizing's happy
#:     path: a concentrated book sits above its per-symbol targets almost
#:     everywhere, and `cap_intent` then refuses every add.
#:   * THREE themes are over target (Cash-generative megacap, Health care,
#:     Defensive) and FOUR are under (both AI sleeves, Core index, Industrials),
#:     so the table shows trims and adds rather than one direction.
#:   * there are exactly 15 positions against a max_positions of 15. Adding to a
#:     held symbol is allowed; opening a new one is not -- which is what makes
#:     the two AI underweights uncloseable and the decision interesting.
#:   * Industrials & energy transition is held at ZERO against a 3% target, so
#:     at least one theme is entirely absent from the book. That is the case
#:     where the trader has to choose between freeing a slot and doing nothing.
#:
#: Gross lands near 64% of equity with ~36% cash -- comfortably inside the 90%
#: effective gross cap that the 10% cash floor implies, and leaving room for the
#: buying-power clamp to be reachable rather than already breached.
SEED_BOOK: dict[str, int] = {
    # AI semiconductors -- underweight as a theme, and holding both capped names
    "NVDA": 105,
    "AVGO": 40,
    "AMD": 19,
    "TSM": 11,
    # AI platforms & applications -- underweight
    "GOOGL": 36,
    "AMZN": 39,
    "META": 14,
    # Cash-generative megacap -- overweight, and MSFT is at the cap
    "MSFT": 47,
    "AAPL": 22,
    # Health care -- overweight
    "LLY": 6,
    "UNH": 20,
    "JNJ": 19,
    # Defensive / uncorrelated -- overweight
    "XOM": 46,
    "JPM": 15,
    # Core index -- underweight
    "QQQ": 7,
}


def _cached_bars() -> dict[str, list[dict[str, Any]]]:
    """``{symbol: bars}`` from every cached ``daily_bars`` payload."""
    if not CACHE_DIR.exists():
        raise SystemExit(
            f"no bars cache at {CACHE_DIR}. Run `make bars` with the Gateway "
            "up first -- this script reads the cache and never fetches."
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(CACHE_DIR.glob("daily_bars-*.json")):
        entry = json.loads(path.read_text())
        symbol = (entry.get("kwargs") or {}).get("symbol")
        bars = (entry.get("payload") or {}).get("bars") or []
        if symbol and bars:
            out[symbol.upper()] = bars
    return out


def _mark(bars: list[dict[str, Any]]) -> tuple[float, date]:
    last = bars[-1]
    return float(last["close"]), date.fromisoformat(last["day"])


def _cost_basis(bars: list[dict[str, Any]]) -> float:
    index = max(len(bars) - COST_BASIS_LOOKBACK_BARS, 0)
    return float(bars[index]["close"])


def seed() -> PortfolioSnapshot:
    cache = _cached_bars()
    missing = sorted(set(SEED_BOOK) - set(cache))
    if missing:
        raise SystemExit(
            f"no cached bars for {missing}. Run `make bars` for the universe "
            "before seeding; the seed prices real closes and will not invent one."
        )

    positions: list[Position] = []
    for symbol, quantity in SEED_BOOK.items():
        bars = cache[symbol]
        price, marked_on = _mark(bars)
        positions.append(Position(
            symbol=symbol, quantity=quantity,
            avg_cost=_cost_basis(bars), last_price=price, marked_on=marked_on,
        ))

    held_value = sum(p.market_value for p in positions)
    cash = round(TARGET_EQUITY - held_value, 2)
    if cash < 0:
        raise SystemExit(
            f"seeded positions are worth {held_value:,.0f} against a target "
            f"equity of {TARGET_EQUITY:,.0f}. Lower the share counts in "
            "SEED_BOOK rather than running a negative cash balance."
        )

    as_of = min(p.marked_on for p in positions)
    return PortfolioSnapshot(
        as_of=as_of, cash=cash, positions=positions, source="seed",
        note=(
            "Seeded for the Stage 6 exit gate. Share counts are hand-chosen to "
            "put the book at max_positions with two symbols at max_position_pct, "
            "one theme underweight and one overweight; every price is a real "
            "cached close. Regenerate with `make portfolio-seed`."
        ),
    )


def refresh(existing: PortfolioSnapshot) -> PortfolioSnapshot:
    cache = _cached_bars()
    positions: list[Position] = []
    for held in existing.positions:
        bars = cache.get(held.symbol)
        if not bars:
            print(f"  {held.symbol}: no cached bars, keeping the existing mark")
            positions.append(held)
            continue
        price, marked_on = _mark(bars)
        positions.append(held.model_copy(update={
            "last_price": price, "marked_on": marked_on,
        }))

    return existing.model_copy(update={
        "positions": positions,
        # The OLDEST mark, so a refresh against a stale cache does not reset
        # the staleness clock. See the module docstring.
        "as_of": min(p.marked_on for p in positions) if positions else existing.as_of,
        "source": "cache",
    })


HEADER = """\
# The book. **State, not configuration** -- see intent/engine.py:load_portfolio
# for why it lives in config/ and is deliberately absent from `config_hash`.
#
# Stage 6 sizing needs equity and the current weight of the symbol under
# consideration, and the exit gate requires ZERO IB CONTACT, so both come from
# this file. `equity` is NOT stored: it is cash + the net value of what is held,
# the way IB reports NetLiquidation. A stored equity that disagrees with the
# positions is a silently wrong position size, so it cannot be stored.
#
# `as_of` is the date of the MARKS, not the day this file was written. Marks
# older than `stale_after_days` are a BLOCKING compliance violation -- sizing
# against last week's weights is how a position gets doubled.
#
# DO NOT HAND-EDIT. Two commands maintain it, neither of which touches IB:
#
#   make portfolio-seed       rebuild the Stage 6 gate's book from the bars cache
#   make portfolio-refresh    re-mark the existing book against the newest bars
#
# At Stage 7, `execute` overwrites this file from the live paper account and
# `source:` becomes `ib`.
"""


def write(snapshot: PortfolioSnapshot, path: Path) -> None:
    import yaml

    body = yaml.safe_dump(
        snapshot.to_yaml_dict(), sort_keys=False, default_flow_style=False
    )
    path.write_text(HEADER + "\n" + body)


def report(snapshot: PortfolioSnapshot) -> None:
    print(f"\nbook as_of {snapshot.as_of}  source={snapshot.source}")
    print(f"  equity   {snapshot.equity:>12,.2f}")
    print(f"  cash     {snapshot.cash:>12,.2f}   {snapshot.cash_pct:5.2f}%")
    print(f"  gross    {snapshot.gross_value:>12,.2f}   "
          f"{snapshot.gross_exposure_pct:5.2f}%")
    print(f"  positions {snapshot.position_count}")
    print()
    print(f"  {'symbol':7} {'qty':>6} {'avg cost':>10} {'mark':>10} "
          f"{'value':>12} {'weight':>7} {'unreal':>8}")
    for p in sorted(snapshot.positions, key=lambda x: -x.market_value):
        pnl = p.unrealized_pct
        print(f"  {p.symbol:7} {p.quantity:6.0f} {p.avg_cost:10,.2f} "
              f"{p.last_price:10,.2f} {p.market_value:12,.2f} "
              f"{snapshot.weight_pct(p.symbol):6.2f}% "
              f"{'' if pnl is None else f'{pnl:+7.1f}%'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", action="store_true",
                       help="rebuild the gate's book from SEED_BOOK and the bars cache")
    group.add_argument("--refresh", action="store_true",
                       help="re-mark the existing book against the newest cached bars")
    group.add_argument("--show", action="store_true", help="print the book and exit")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.show:
        report(PortfolioSnapshot.load(args.out))
        return 0

    snapshot = seed() if args.seed else refresh(PortfolioSnapshot.load(args.out))
    write(snapshot, args.out)
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    report(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
