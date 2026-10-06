#!/usr/bin/env python3
"""Fetch daily bars from IB into the provider cache. Runs on the EXECUTE side.

This is a separate process on purpose. It imports ``ib_async`` (via
``research_desk.execution.bars``), which ``decide`` must never do -- see §0 and
``providers/ib.py``. Keeping it out of the ``desk`` CLI is what makes that
guarantee structural rather than aspirational.

    make bars                       # the whole universe + benchmark + sectors
    make bars SYMBOLS="AAPL MSFT"
    python scripts/fetch_bars.py --symbols AAPL --duration "2 Y"

Needs the Gateway up (``make gateway-start``) and IB_USERNAME/IB_PASSWORD set.
Run it whenever you want fresher prices; ``decide`` reads whatever is cached
and never needs the Gateway itself.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_desk.config import (  # noqa: E402
    ConfigError,
    load_settings,
    load_yaml,
    resolve_ib_endpoint,
)
from research_desk.execution.bars import DEFAULT_DURATION, fetch_and_cache  # noqa: E402
from research_desk.logging_setup import setup_logging  # noqa: E402
from research_desk.providers.cache import Cache  # noqa: E402


def provider_symbol_map() -> dict[str, dict[str, str]]:
    """symbol -> {provider: spelling}, from universe.yaml.

    Only BRK.B needs one today, but getting it wrong is a silently empty
    series rather than an error, so it is read rather than assumed.
    """
    body = load_yaml("universe.yaml")
    return {
        entry["symbol"].upper(): entry["provider_symbols"]
        for entry in body["symbols"]
        if entry.get("provider_symbols")
    }


def universe_symbols() -> list[str]:
    """Everything that needs bars: the universe, its benchmark, its sector ETFs.

    The sector ETFs are easy to forget and §7.1 needs them: relative strength
    is measured against SPY *and* the sector, and "absolute return tells you
    almost nothing in a trending market". A missing sector series is a missing
    metric, not a cosmetic gap.
    """
    body = load_yaml("universe.yaml")
    symbols = {entry["symbol"] for entry in body["symbols"]}
    symbols.add(body["benchmark"])
    symbols.update(
        entry["sector_etf"] for entry in body["symbols"] if entry.get("sector_etf")
    )
    return sorted(symbols)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="symbols to fetch (default: the whole universe)")
    parser.add_argument("--duration", default=DEFAULT_DURATION,
                        help=f"IB duration string (default {DEFAULT_DURATION!r})")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: latest)")
    args = parser.parse_args(argv)

    setup_logging()
    settings = load_settings()
    symbols = [s.upper() for s in (args.symbols or universe_symbols())]
    end = date.fromisoformat(args.end) if args.end else None

    # This script runs on the HOST, so the container hostname in .env will not
    # resolve. Probe both and say which one answered.
    try:
        host, port = resolve_ib_endpoint(settings)
    except ConfigError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Fetching {len(symbols)} symbol(s) from {host}:{port}")
    print(f"  {', '.join(symbols)}\n")

    try:
        written = asyncio.run(fetch_and_cache(
            symbols,
            Cache(settings.cache_dir),
            host=host,
            port=port,
            # 12, not 11: 11 is `execute`'s slot and a clientId collision does
            # not error, it silently fails to connect (migration plan §0).
            client_id=12,
            duration=args.duration,
            end=end,
            provider_symbols=provider_symbol_map(),
        ))
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        print("\nIs the Gateway up and logged in?  make gateway-logs", file=sys.stderr)
        return 1

    print()
    for symbol, count in sorted(written.items()):
        print(f"  {symbol:<8} {count:>5} bars")
    print(f"\nCache: {settings.cache_dir / 'ib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
