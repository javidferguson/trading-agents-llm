#!/usr/bin/env python3
"""Standalone GDELT tone and coverage-volume collector.

WHY THIS EXISTS BEFORE THE PROVIDER LAYER
=========================================
GDELT's open API serves a **rolling ~3-month window**. Every day not cached now
is a day permanently absent from the Stage 8 evaluation, and no amount of later
effort recovers it. It is the only task in the whole plan whose cost goes *up*
the longer it is deferred (architecture §7.5, migration plan Stage 2).

So this script is deliberately **decoupled from everything**:

* It imports nothing from ``research_desk``. It is not the provider layer, and
  ``providers/gdelt.py`` at Stage 2 will read this cache rather than replace it.
* Its only dependency beyond the standard library is PyYAML, to read the
  universe.
* It is idempotent. Run it as often as you like; re-running a day overwrites
  that day's record and nothing else.

WHAT IT DOES NOT DO
===================
It does not compute percentiles or z-scores. Those are ``indicators.py``'s job
at Stage 2, because all arithmetic lives there (architecture §2). This script
collects raw daily tone and raw daily article counts. Nothing else.

THE ONE THING TO DO BY HAND, ONCE
=================================
GDELT matches **text, not tickers**. Run::

    python scripts/gdelt_collect.py --inspect META

and read the headlines. If they are about metaphysics rather than the company,
fix ``gdelt_query`` in ``config/universe.yaml``. A wrong query produces a tone
series that measures something else entirely, which is worse than no series at
all because it looks like data.

USAGE
=====
::

    python scripts/gdelt_collect.py                  # last 7 days, all symbols
    python scripts/gdelt_collect.py --days 90        # first-run backfill
    python scripts/gdelt_collect.py --symbols AAPL,MSFT
    python scripts/gdelt_collect.py --status         # coverage and gaps
    python scripts/gdelt_collect.py --inspect META   # eyeball the query
    python scripts/gdelt_collect.py --probe-window   # how far back does it go?

Once the output looks right, put it behind a daily cron. Until then, running it
by hand each day is enough -- the point is only that the days accumulate.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = REPO_ROOT / "config" / "universe.yaml"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "gdelt"

API = "https://api.gdeltproject.org/api/v2/doc/doc"

#: GDELT is an open API with no key and no published quota, which is not the
#: same as no limit. One request per second, and we make two per symbol.
REQUEST_INTERVAL_S = 1.5
TIMEOUT_S = 45
USER_AGENT = "research-desk/0.1 (gdelt sentiment cache; contact via repo)"


class GdeltError(RuntimeError):
    """The API returned something we cannot use."""


def _ssl_context() -> ssl.SSLContext:
    """A context with a CA bundle that actually exists.

    python.org's macOS builds ship without a usable CA store unless you have run
    "Install Certificates.command", so stdlib ``urllib`` fails TLS with
    ``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`` while
    ``curl`` on the same machine is fine. ``certifi`` is already present in the
    venv (httpx and requests both pull it), so prefer it and fall back to the
    system store when running under a bare interpreter.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _get(params: dict[str, str]) -> Any:
    """One GET, returning parsed JSON.

    GDELT answers a malformed or over-broad query with an HTTP 200 carrying a
    plain-text error, so a status check alone is not enough -- the JSON parse
    failure is the real signal and it gets reported with the body attached.
    """
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S, context=_SSL) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise GdeltError(f"HTTP {exc.code} for {params.get('mode')}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise GdeltError(f"Network error for {params.get('mode')}: {exc.reason}") from exc

    stripped = body.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise GdeltError(f"Non-JSON reply for {params.get('mode')}: {stripped[:200]!r}") from exc


def _stamp(day: date, end: bool = False) -> str:
    return day.strftime("%Y%m%d") + ("235959" if end else "000000")


def _series(payload: Any, wanted: str) -> dict[str, float]:
    """Pull one named series out of a GDELT timeline response, keyed by ISO date."""
    out: dict[str, float] = {}
    for series in (payload or {}).get("timeline", []):
        name = (series.get("series") or "").lower()
        if wanted not in name:
            continue
        for point in series.get("data", []):
            raw = point.get("date")
            if not raw:
                continue
            # GDELT returns "2026-08-01T00:00:00Z" here and "20260801000000"
            # elsewhere. Accept both rather than guessing which endpoint changed.
            try:
                if "T" in raw:
                    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
                else:
                    parsed = datetime.strptime(raw[:8], "%Y%m%d").date()
            except ValueError:
                continue
            out[parsed.isoformat()] = float(point.get("value", 0.0))
    return out


# --------------------------------------------------------------------------- #
# Universe and store
# --------------------------------------------------------------------------- #


def load_universe(only: list[str] | None = None) -> list[dict[str, Any]]:
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Missing {UNIVERSE_FILE}")
    body = yaml.safe_load(UNIVERSE_FILE.read_text()) or {}
    symbols = body.get("symbols") or []

    missing = [s["symbol"] for s in symbols if not s.get("gdelt_query")]
    if missing:
        print(f"warn: no gdelt_query for {', '.join(missing)}; skipping them", file=sys.stderr)
    symbols = [s for s in symbols if s.get("gdelt_query")]

    if only:
        wanted = {s.strip().upper() for s in only}
        found = {s["symbol"].upper() for s in symbols}
        for unknown in wanted - found:
            print(f"warn: {unknown} is not in universe.yaml", file=sys.stderr)
        symbols = [s for s in symbols if s["symbol"].upper() in wanted]
    return symbols


def store_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def read_store(symbol: str) -> dict[str, Any]:
    path = store_path(symbol)
    if not path.exists():
        return {"symbol": symbol.upper(), "query": None, "days": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A truncated write must not cost the whole history. Keep the damaged
        # file for inspection and start a new one.
        salvage = path.with_suffix(".json.corrupt")
        path.rename(salvage)
        print(f"warn: {path.name} was unreadable; moved to {salvage.name}", file=sys.stderr)
        return {"symbol": symbol.upper(), "query": None, "days": {}}


def write_store(symbol: str, store: dict[str, Any]) -> None:
    """Write atomically. A partial file here is three months of lost history."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = store_path(symbol)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=1, sort_keys=True))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #


def collect_symbol(entry: dict[str, Any], start: date, end: date, keep_raw: bool) -> int:
    """Fetch and merge one symbol's window. Returns how many days were recorded."""
    symbol = entry["symbol"].upper()
    query = entry["gdelt_query"]

    base = {
        "query": query,
        "format": "json",
        "startdatetime": _stamp(start),
        "enddatetime": _stamp(end, end=True),
    }

    tone_raw = _get({**base, "mode": "timelinetone"})
    time.sleep(REQUEST_INTERVAL_S)
    volume_raw = _get({**base, "mode": "timelinevolraw"})

    tone = _series(tone_raw, "tone")
    counts = _series(volume_raw, "article count")
    monitored = _series(volume_raw, "total monitored")

    store = read_store(symbol)
    if store.get("query") and store["query"] != query:
        # The query defines what the series measures. Changing it silently would
        # splice two different measurements into one history and nothing
        # downstream could tell.
        print(
            f"warn: {symbol} query changed.\n"
            f"      was: {store['query']}\n"
            f"      now: {query}\n"
            f"      Days before today were collected under the OLD query.",
            file=sys.stderr,
        )
    store["query"] = query
    store.setdefault("days", {})

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recorded = 0
    day = start
    while day <= end:
        key = day.isoformat()
        # A day inside the requested window that GDELT did not return is
        # genuinely zero coverage, not a missing fetch. Recording it explicitly
        # is what lets Stage 2 tell "no articles" from "never collected" -- and
        # architecture §7.5 is emphatic that the second must never be silently
        # rendered as the first.
        store["days"][key] = {
            "tone": tone.get(key),
            "articles": int(counts.get(key, 0)),
            "total_monitored": int(monitored[key]) if key in monitored else None,
            "observed": True,
            "fetched_at": fetched_at,
        }
        recorded += 1
        day += timedelta(days=1)

    if keep_raw:
        raw_dir = CACHE_DIR / "raw" / symbol
        raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{start.isoformat()}_{end.isoformat()}"
        (raw_dir / f"{stem}-tone.json").write_text(json.dumps(tone_raw))
        (raw_dir / f"{stem}-volraw.json").write_text(json.dumps(volume_raw))

    write_store(symbol, store)
    return recorded


def cmd_collect(args: argparse.Namespace) -> int:
    end = date.today() - timedelta(days=1)   # today is still accumulating
    start = end - timedelta(days=args.days - 1)
    universe = load_universe(args.symbols.split(",") if args.symbols else None)
    if not universe:
        print("Nothing to collect.", file=sys.stderr)
        return 1

    print(f"GDELT {start} .. {end}  ({args.days} days, {len(universe)} symbols)\n")
    failures = 0
    for index, entry in enumerate(universe):
        symbol = entry["symbol"]
        try:
            recorded = collect_symbol(entry, start, end, keep_raw=args.keep_raw)
        except GdeltError as exc:
            print(f"  {symbol:<6} FAILED  {exc}", file=sys.stderr)
            failures += 1
        else:
            store = read_store(symbol)
            window = {d: store["days"][d] for d in store["days"] if start.isoformat() <= d <= end.isoformat()}
            articles = sum(d["articles"] for d in window.values())
            toned = [d["tone"] for d in window.values() if d["tone"] is not None]
            mean = f"{sum(toned) / len(toned):+.2f}" if toned else "  n/a"
            print(f"  {symbol:<6} {recorded:>3}d  {articles:>6} articles  mean tone {mean}")
        if index < len(universe) - 1:
            time.sleep(REQUEST_INTERVAL_S)

    print(f"\nCache: {CACHE_DIR}")
    if failures:
        print(f"{failures} symbol(s) failed.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def cmd_status(args: argparse.Namespace) -> int:
    """Coverage per symbol, and where the gaps are.

    The gap count is the number that matters. A gap is a day that is gone for
    good once it falls out of GDELT's rolling window.
    """
    universe = load_universe(args.symbols.split(",") if args.symbols else None)
    if not universe:
        return 1

    print(f"{'symbol':<8}{'first':<12}{'last':<12}{'days':>6}{'gaps':>6}  {'articles':>9}")
    print("-" * 60)

    total_gaps = 0
    for entry in universe:
        symbol = entry["symbol"]
        days = read_store(symbol).get("days", {})
        if not days:
            print(f"{symbol:<8}{'-':<12}{'-':<12}{0:>6}{'-':>6}  {'-':>9}")
            continue

        keys = sorted(days)
        first, last = date.fromisoformat(keys[0]), date.fromisoformat(keys[-1])
        span = (last - first).days + 1
        gaps = span - len(keys)
        total_gaps += gaps
        articles = sum(d.get("articles", 0) for d in days.values())
        print(f"{symbol:<8}{keys[0]:<12}{keys[-1]:<12}{len(keys):>6}{gaps:>6}  {articles:>9,}")

    print()
    latest = [max(read_store(e["symbol"]).get("days", {}), default="") for e in universe]
    collected = [k for k in latest if k]

    if not collected:
        # Distinct from "behind": there is no history at all, and the clock on
        # the rolling window is already running.
        print("No history collected yet. Start with a backfill:")
        print("    python scripts/gdelt_collect.py --days 90")
        print("Every day not collected now is a day permanently absent from the")
        print("Stage 8 evaluation.")
        return 0

    if total_gaps:
        print(f"{total_gaps} missing day(s). Re-collect with --days N while they are still")
        print("inside GDELT's rolling window; after that they are unrecoverable.")
    else:
        print("No gaps.")

    stale = date.today() - timedelta(days=2)
    if all(date.fromisoformat(k) < stale for k in collected):
        print("\nEvery symbol is more than 2 days behind. Run `make gdelt`.")
    elif len(collected) < len(universe):
        behind = len(universe) - len(collected)
        print(f"\n{behind} symbol(s) have no history at all. Run `make gdelt`.")
    return 0


# --------------------------------------------------------------------------- #
# Inspect -- the hand-check the universe file asks for
# --------------------------------------------------------------------------- #


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print recent headlines for one symbol's query, so you can eyeball it."""
    matches = load_universe([args.inspect])
    if not matches:
        print(f"{args.inspect} is not in universe.yaml", file=sys.stderr)
        return 1
    entry = matches[0]

    payload = _get({
        "query": entry["gdelt_query"],
        "mode": "artlist",
        "format": "json",
        "maxrecords": "25",
        "sort": "hybridrel",
        "timespan": "7d",
    })
    articles = (payload or {}).get("articles", [])

    print(f"{entry['symbol']} -- {entry['name']}")
    print(f"query: {entry['gdelt_query']}\n")
    if not articles:
        print("NO ARTICLES. The query is too narrow, or malformed.")
        return 1

    for article in articles:
        domain = (article.get("domain") or "?")[:24]
        title = (article.get("title") or "").strip()[:96]
        print(f"  {domain:<26} {title}")

    print(f"\n{len(articles)} article(s). Are these about the company?")
    print("If not, fix gdelt_query in config/universe.yaml before collecting more.")
    return 0


# --------------------------------------------------------------------------- #
# Probe -- resolve the open question in architecture §7.5
# --------------------------------------------------------------------------- #


def cmd_probe(args: argparse.Namespace) -> int:
    """Measure how far back the API actually serves.

    Architecture §7.5 flags a contradiction it could not resolve: GDELT's
    announcement says the API searches "a rolling window of the last 3 months"
    while also referring to an index reaching back to 2017-01-01. The doc asks
    for this to be measured rather than trusted. This measures it.
    """
    probe_query = '"stock market"'
    offsets = [7, 30, 60, 90, 120, 180, 365, 730, 1825, 3650]

    print(f"Probing with query {probe_query}, one-day windows.\n")
    print(f"{'days back':>10}  {'date':<12}  result")
    print("-" * 46)

    deepest: int | None = None
    for offset in offsets:
        day = date.today() - timedelta(days=offset)
        try:
            payload = _get({
                "query": probe_query,
                "mode": "timelinevolraw",
                "format": "json",
                "startdatetime": _stamp(day),
                "enddatetime": _stamp(day, end=True),
            })
            points = _series(payload, "article count")
            if points:
                deepest = offset
                print(f"{offset:>10}  {day}  ok, {len(points)} point(s)")
            else:
                print(f"{offset:>10}  {day}  empty")
        except GdeltError as exc:
            print(f"{offset:>10}  {day}  {exc}")
        time.sleep(REQUEST_INTERVAL_S)

    print()
    if deepest is None:
        print("Nothing returned at any depth -- check connectivity before concluding.")
        return 1
    print(f"Deepest window returning data: ~{deepest} days (~{deepest / 30:.1f} months).")
    print("Record this in architecture §7.5, which currently states the question")
    print("as unresolved. If it really is ~90 days, caching daily is mandatory.")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=7,
                        help="how many days back to collect (default 7; use 90 for a first backfill)")
    parser.add_argument("--symbols", default=None, help="comma-separated subset")
    parser.add_argument("--keep-raw", action="store_true", default=True,
                        help="also write the raw API responses (default on)")
    parser.add_argument("--no-keep-raw", dest="keep_raw", action="store_false")
    parser.add_argument("--status", action="store_true", help="report coverage and gaps, fetch nothing")
    parser.add_argument("--inspect", metavar="SYMBOL", help="print recent headlines for one query")
    parser.add_argument("--probe-window", action="store_true",
                        help="measure how far back the API serves")
    args = parser.parse_args(argv)

    if args.status:
        return cmd_status(args)

    # --status is the only mode that touches no network. For the rest, an
    # unreachable API is an ordinary outcome -- GDELT is a free service with no
    # uptime commitment -- and it deserves one line, not a traceback. Getting
    # this wrong matters more than usual once this runs from cron, where a
    # traceback is just a mail nobody reads.
    try:
        if args.inspect:
            return cmd_inspect(args)
        if args.probe_window:
            return cmd_probe(args)
        return cmd_collect(args)
    except GdeltError as exc:
        print(f"GDELT unavailable: {exc}", file=sys.stderr)
        print("Nothing was written. Re-run while the day is still inside the", file=sys.stderr)
        print("rolling window -- `--status` shows what is missing.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; partial days are already written.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
