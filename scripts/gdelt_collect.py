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

#: GDELT's own rate limit, quoted verbatim from the body of a 429 it returned:
#:
#:     "Please limit requests to one every 5 seconds or contact
#:      kalev.leetaru5@gmail.com for larger queries. All high-traffic users
#:      should switch to our ngrams dataset..."
#:
#: This is not a guess and it is not negotiable by being polite about it. An
#: earlier value of 1.5s here was over three times too fast and got the IP
#: throttled, which then looks exactly like the API being down.
#:
#: **7, not 5, and the margin is the point.** 5.0 is exactly the published
#: limit, and `_throttle` gates on request START times -- so GDELT receives
#: requests 5.00s apart with network jitter on top, and some of those arrivals
#: land at 4.98s. Pacing precisely at a threshold means breaching it regularly,
#: and a 429 then triggers retries that deepen the block. Measured 2026-10-08:
#: an IP was still refused ~18 hours after being throttled.
#:
#: The cost of the margin is small and the cost of a block is days: 32 symbols
#: x 2 requests goes from 5.3 to 7.5 minutes. Lower it with `--interval` if you
#: are collecting one symbol and in a hurry; do not lower the default.
REQUEST_INTERVAL_S = 7.0
TIMEOUT_S = 45
USER_AGENT = "research-desk/0.1 (gdelt sentiment cache; contact via repo)"

#: Backoff schedule for 429 and 5xx. GDELT sends NO Retry-After header (checked
#: against the live API), so the schedule has to be ours.
#:
#: DELIBERATELY SHORT, and this is the counter-intuitive part. GDELT's 429 has
#: two quite different causes:
#:
#:   * a momentary burst -- you went faster than one request / 5s. Clears in
#:     seconds, and one retry fixes it.
#:   * a SUSTAINED per-IP block, earned by sustained over-use. Measured: an IP
#:     in this state was still refused after 150s of backoff across four
#:     retries. It is a penalty box, not a rolling window.
#:
#: Retrying hard cannot distinguish them, and in the second case every extra
#: request is more traffic from an IP that is being punished for traffic --
#: plausibly extending the block. So: two retries, ~40s. If the second fails we
#: are in the second case, and the correct move is to stop and wait, which is
#: what the error then says.
RETRY_BACKOFF_S = (10.0, 30.0)

#: Status codes worth trying again. 429 is the throttle; 5xx is GDELT having a
#: moment. Everything else (404, 400) is our bug and retrying just hides it.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Has any request in THIS process been accepted? Distinguishes "we went too
#: fast" from "this IP was already blocked", which need opposite responses:
#: back off, versus stop entirely.
_succeeded_once = False

#: Monotonic timestamp of the last request, for the inter-request gate below.
_last_request_at: float | None = None


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


def _throttle() -> None:
    """Block until at least ``REQUEST_INTERVAL_S`` has passed since the last request.

    A clock gate rather than a trailing ``sleep()`` after each call. The
    difference matters: a trailing sleep ignores how long the request itself
    took, and it leaves the gap between two back-to-back requests dependent on
    every call site remembering to sleep. Gating on a monotonic timestamp makes
    "never less than 5s apart" true by construction, wherever ``_get`` is
    called from.

    ``time.monotonic`` rather than ``time.time`` so that an NTP correction or a
    DST change cannot produce a negative interval and let a burst through.
    """
    global _last_request_at
    if _last_request_at is not None:
        waited = time.monotonic() - _last_request_at
        remaining = REQUEST_INTERVAL_S - waited
        if remaining > 0:
            time.sleep(remaining)
    _last_request_at = time.monotonic()


def _describe(body: str, limit: int = 300) -> str:
    """Collapse a server message to one readable line."""
    return " ".join(body.split())[:limit]


def _get(params: dict[str, str]) -> Any:
    """One GET, returning parsed JSON. Rate-limited and retried.

    Three failure shapes, all of which GDELT actually produces:

    1. **HTTP 429 with the explanation in the body.** The body is the only place
       the rate limit is stated -- there is no Retry-After header. An earlier
       version of this function formatted ``exc.reason`` ("Too Many Requests")
       and discarded ``exc.read()``, so the API told us exactly what was wrong
       and we printed a generic status line. ``HTTPError`` is file-like; read it.
    2. **HTTP 200 with a plain-text error**, for a malformed or over-broad
       query. A status check alone will not catch this; the JSON parse failure
       is the real signal, so the body is attached to the error.
    3. **Transient network errors**, which get the same backoff as 429.
    """
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    mode = params.get("mode", "?")

    # Whether THIS process has ever had a request accepted. A 429 before that
    # cannot have been caused by our own pacing, so it is a pre-existing block
    # and retrying is pure harm -- see the 429 branch below.
    global _succeeded_once

    last_error = ""
    for attempt in range(len(RETRY_BACKOFF_S) + 1):
        _throttle()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S, context=_SSL) as response:
                body = response.read().decode("utf-8", errors="replace")
            _succeeded_once = True
            break
        except urllib.error.HTTPError as exc:
            # The body carries GDELT's own words. Read it before deciding
            # anything, including whether to retry.
            try:
                detail = _describe(exc.read().decode("utf-8", errors="replace"))
            except Exception:
                detail = exc.reason or ""
            last_error = f"HTTP {exc.code} for {mode}: {detail or exc.reason}"
            # A 429 before this process has had ANY request accepted is a
            # block that was already in place. Our pacing cannot have caused
            # it, so the backoff cannot clear it -- and each retry is one more
            # request from an IP being punished for requests. Fail immediately.
            if exc.code == 429 and not _succeeded_once:
                raise GdeltError(
                    f"{last_error}\n"
                    "  REFUSED ON THE FIRST REQUEST, so this IP was ALREADY "
                    "blocked before this run started.\n"
                    "  Not retrying: our pacing did not cause it, so backoff "
                    "cannot clear it, and every\n"
                    "  further request is more traffic from an IP being "
                    "punished for traffic.\n"
                    "\n"
                    "  These blocks are long. Measured 2026-10-08: still "
                    "refused ~18 hours later.\n"
                    "  What actually helps, in order:\n"
                    "    * wait hours, not minutes, and probe SPARINGLY -- "
                    "once an hour at most\n"
                    "    * run it from a different network; the block is "
                    "per-IP\n"
                    "    * mail kalev.leetaru5@gmail.com, which GDELT's own "
                    "429 invites for this volume\n"
                    "\n"
                    "  Nothing was lost. `--status` shows what is missing, and "
                    "recent days stay\n"
                    "  reachable for ~3 months -- waiting costs depth at the "
                    "far end, not today."
                ) from exc

            if exc.code not in RETRYABLE_STATUS or attempt >= len(RETRY_BACKOFF_S):
                if exc.code == 429:
                    raise GdeltError(
                        f"{last_error}\n"
                        "  This survived the full backoff, so it is a SUSTAINED "
                        "per-IP block rather than a momentary burst.\n"
                        "  Retrying now adds traffic from an IP already being "
                        "throttled and may extend it.\n"
                        "  Stop for several HOURS, not minutes, then re-run. "
                        "Nothing was lost --\n"
                        "  `--status` shows what is still missing."
                    ) from exc
                raise GdeltError(last_error) from exc
        except urllib.error.URLError as exc:
            last_error = f"Network error for {mode}: {exc.reason}"
            if attempt >= len(RETRY_BACKOFF_S):
                raise GdeltError(last_error) from exc

        pause = RETRY_BACKOFF_S[attempt]
        print(
            f"    {last_error}\n"
            f"    retrying in {pause:.0f}s "
            f"({attempt + 1}/{len(RETRY_BACKOFF_S)})",
            file=sys.stderr,
        )
        time.sleep(pause)
    else:  # pragma: no cover - the loop always breaks or raises
        raise GdeltError(last_error or f"Gave up on {mode}")

    stripped = body.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise GdeltError(f"Non-JSON reply for {mode}: {_describe(stripped, 200)!r}") from exc


def _stamp(day: date, end: bool = False) -> str:
    return day.strftime("%Y%m%d") + ("235959" if end else "000000")


def _has_timeline(payload: Any) -> bool:
    """Did GDELT actually answer, or just hand back something contentless?

    This is the difference between "no articles that day" and "no reply", and
    conflating them is the single most damaging thing this script could do.

    Observed in the wild: GDELT returns **HTTP 200 with an empty body** when it
    is unhappy -- still throttled, or the query produced nothing it wants to
    discuss. An earlier version parsed that into zero series, found no data for
    any day in the window, and dutifully recorded `articles: 0, observed: True`
    for every one of them. Those days then look COLLECTED: `--status` shows no
    gap, nothing ever re-fetches them, and ~3 months later the real numbers are
    unrecoverable. A false zero is worse than a hole, because a hole is visible.

    So: a payload is an observation only if it carries a `timeline` with at
    least one series. Anything else is a failed fetch and must not be written.
    """
    if not isinstance(payload, dict):
        return False
    timeline = payload.get("timeline")
    return isinstance(timeline, list) and len(timeline) > 0


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

    # No sleep between these two: _get() gates on the clock, so the 5s gap is
    # enforced whether or not a caller remembers to wait.
    tone_raw = _get({**base, "mode": "timelinetone"})
    volume_raw = _get({**base, "mode": "timelinevolraw"})

    # Refuse to record anything from a contentless reply. See _has_timeline:
    # writing observed-zero here would permanently fake "no coverage" for every
    # day in the window, and the window cannot be re-fetched later.
    if not _has_timeline(volume_raw):
        raise GdeltError(
            f"{symbol}: GDELT returned no timeline for "
            f"{start}..{end} (HTTP 200, empty or unrecognised body). "
            "Refusing to record -- writing zeros here would look like 'no "
            "coverage' forever. Left as a gap; `--status` will show it."
        )

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

    # At 5s/request and two requests per symbol, this run takes at least
    # 2 * 5 * len(universe) seconds. Say so up front, and print each symbol
    # BEFORE fetching it -- otherwise correct, patient behaviour is
    # indistinguishable from a hang.
    floor_s = 2 * REQUEST_INTERVAL_S * len(universe)
    print(f"GDELT {start} .. {end}  ({args.days} days, {len(universe)} symbols)")
    print(f"Rate limit is one request / {REQUEST_INTERVAL_S:.0f}s, so this takes "
          f">= {floor_s / 60:.1f} min.\n")

    failures = 0
    for index, entry in enumerate(universe):
        symbol = entry["symbol"]
        print(f"  {symbol:<6} [{index + 1}/{len(universe)}] fetching...", flush=True)
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

    print(f"Probing with query {probe_query}, one-day windows.")
    print(f"{len(offsets)} requests at one / {REQUEST_INTERVAL_S:.0f}s "
          f"= ~{len(offsets) * REQUEST_INTERVAL_S / 60:.1f} min.\n")
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
    # Module-level because _get()'s clock gate reads it, and _get() is reached
    # from four different commands. Declared up here because the argparse
    # defaults below already reference the name.
    global REQUEST_INTERVAL_S

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
    parser.add_argument("--interval", type=float, default=REQUEST_INTERVAL_S,
                        help=f"seconds between requests (default {REQUEST_INTERVAL_S:.0f}; "
                             "GDELT asks for one every 5s -- lowering this gets you a 429)")
    args = parser.parse_args(argv)
    REQUEST_INTERVAL_S = args.interval

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
