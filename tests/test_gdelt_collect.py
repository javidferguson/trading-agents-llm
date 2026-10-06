"""Offline tests for the GDELT collector.

The script's whole job is to accumulate days that cannot be re-fetched later, so
its merge and parse logic needs to be right the first time. These stub the
transport rather than hitting the API: GDELT is a free service with no uptime
commitment, and a test suite that fails when it is slow is a test suite people
learn to ignore.

It lives in ``scripts/`` rather than in the package on purpose (it must not
depend on ``research_desk``), so it is loaded here by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gdelt_collect.py"


def _load():
    spec = importlib.util.spec_from_file_location("gdelt_collect", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gdelt_collect"] = module
    spec.loader.exec_module(module)
    return module


gdelt = _load()


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gdelt, "CACHE_DIR", tmp_path / "gdelt")
    monkeypatch.setattr(gdelt, "REQUEST_INTERVAL_S", 0)


def _timeline(series_name: str, points: dict[str, float]) -> dict[str, Any]:
    return {
        "timeline": [
            {
                "series": series_name,
                "data": [{"date": f"{d}T00:00:00Z", "value": v} for d, v in points.items()],
            }
        ]
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_series_accepts_both_date_formats() -> None:
    """GDELT returns ISO in one endpoint and YYYYMMDDHHMMSS in another."""
    payload = {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"date": "2026-08-01T00:00:00Z", "value": -1.5},
                    {"date": "20260802000000", "value": 2.25},
                ],
            }
        ]
    }
    assert gdelt._series(payload, "tone") == {"2026-08-01": -1.5, "2026-08-02": 2.25}


def test_series_ignores_other_series_in_the_same_response() -> None:
    payload = {
        "timeline": [
            {"series": "Article Count", "data": [{"date": "2026-08-01T00:00:00Z", "value": 12}]},
            {"series": "Total Monitored Articles", "data": [{"date": "2026-08-01T00:00:00Z", "value": 9e6}]},
        ]
    }
    assert gdelt._series(payload, "article count") == {"2026-08-01": 12.0}
    assert gdelt._series(payload, "total monitored") == {"2026-08-01": 9e6}


def test_series_survives_a_malformed_point() -> None:
    """One bad row must not cost the whole day's fetch."""
    payload = {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"date": "not-a-date", "value": 1.0},
                    {"value": 2.0},
                    {"date": "2026-08-03T00:00:00Z", "value": 3.0},
                ],
            }
        ]
    }
    assert gdelt._series(payload, "tone") == {"2026-08-03": 3.0}


def test_empty_and_missing_payloads_are_not_errors() -> None:
    assert gdelt._series({}, "tone") == {}
    assert gdelt._series(None, "tone") == {}


# --------------------------------------------------------------------------- #
# Collection and merge
# --------------------------------------------------------------------------- #


def _stub_get(monkeypatch, tone: dict, counts: dict, monitored: dict | None = None):
    def fake(params):
        if params["mode"] == "timelinetone":
            return _timeline("Average Tone", tone)
        payload = _timeline("Article Count", counts)
        payload["timeline"].append(
            {
                "series": "Total Monitored Articles",
                "data": [
                    {"date": f"{d}T00:00:00Z", "value": v}
                    for d, v in (monitored or {}).items()
                ],
            }
        )
        return payload

    monkeypatch.setattr(gdelt, "_get", fake)


ENTRY = {"symbol": "AAPL", "name": "Apple Inc.", "gdelt_query": '("Apple Inc")'}


def test_days_without_coverage_are_recorded_as_observed_zero(monkeypatch) -> None:
    """The distinction architecture §7.5 is emphatic about.

    A day inside the requested window that GDELT did not return is genuinely
    zero coverage. A day never requested is unknown. Collapsing the second into
    the first would quietly corrupt the ablation that decides whether sentiment
    earns its place.
    """
    start, end = date(2026, 8, 1), date(2026, 8, 3)
    _stub_get(monkeypatch, tone={"2026-08-01": -1.0}, counts={"2026-08-01": 5})

    assert gdelt.collect_symbol(ENTRY, start, end, keep_raw=False) == 3
    days = gdelt.read_store("AAPL")["days"]

    assert days["2026-08-01"]["tone"] == -1.0
    assert days["2026-08-01"]["articles"] == 5

    # Present in the window, absent from the reply: observed, zero articles,
    # and tone None rather than 0.0 -- a tone of zero means "neutral coverage",
    # which is a different claim entirely.
    assert days["2026-08-02"]["observed"] is True
    assert days["2026-08-02"]["articles"] == 0
    assert days["2026-08-02"]["tone"] is None

    # Never requested at all: simply absent from the store.
    assert "2026-07-31" not in days


def test_re_running_is_idempotent_and_preserves_other_days(monkeypatch) -> None:
    """Runs overlap constantly -- daily cron plus the odd manual backfill."""
    _stub_get(monkeypatch, tone={"2026-08-01": -1.0}, counts={"2026-08-01": 5})
    gdelt.collect_symbol(ENTRY, date(2026, 8, 1), date(2026, 8, 1), keep_raw=False)

    _stub_get(monkeypatch, tone={"2026-08-02": 2.0}, counts={"2026-08-02": 9})
    gdelt.collect_symbol(ENTRY, date(2026, 8, 2), date(2026, 8, 2), keep_raw=False)

    days = gdelt.read_store("AAPL")["days"]
    assert set(days) == {"2026-08-01", "2026-08-02"}
    assert days["2026-08-01"]["articles"] == 5
    assert days["2026-08-02"]["articles"] == 9


def test_changing_the_query_warns_because_it_changes_the_measurement(monkeypatch, capsys) -> None:
    _stub_get(monkeypatch, tone={"2026-08-01": 1.0}, counts={"2026-08-01": 1})
    gdelt.collect_symbol(ENTRY, date(2026, 8, 1), date(2026, 8, 1), keep_raw=False)

    changed = {**ENTRY, "gdelt_query": '("Apple" OR "iPhone")'}
    gdelt.collect_symbol(changed, date(2026, 8, 2), date(2026, 8, 2), keep_raw=False)

    warning = capsys.readouterr().err
    assert "query changed" in warning
    assert "OLD query" in warning


def test_a_corrupt_store_is_set_aside_rather_than_read(monkeypatch) -> None:
    """A truncated write must not be allowed to take the history with it."""
    gdelt.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gdelt.store_path("AAPL").write_text("{ this is not json")

    store = gdelt.read_store("AAPL")
    assert store["days"] == {}
    assert gdelt.store_path("AAPL").with_suffix(".json.corrupt").exists()


def test_writes_are_atomic(monkeypatch) -> None:
    """No .tmp left behind, and the real file is complete JSON."""
    _stub_get(monkeypatch, tone={"2026-08-01": 1.0}, counts={"2026-08-01": 3})
    gdelt.collect_symbol(ENTRY, date(2026, 8, 1), date(2026, 8, 1), keep_raw=False)

    assert not list(gdelt.CACHE_DIR.glob("*.tmp"))
    json.loads(gdelt.store_path("AAPL").read_text())


def test_keep_raw_writes_the_unparsed_responses(monkeypatch) -> None:
    """The raw reply is the honest substrate; parsing can be redone, fetching cannot."""
    _stub_get(monkeypatch, tone={"2026-08-01": 1.0}, counts={"2026-08-01": 3})
    gdelt.collect_symbol(ENTRY, date(2026, 8, 1), date(2026, 8, 1), keep_raw=True)

    raw = sorted(p.name for p in (gdelt.CACHE_DIR / "raw" / "AAPL").iterdir())
    assert raw == ["2026-08-01_2026-08-01-tone.json", "2026-08-01_2026-08-01-volraw.json"]


def test_a_long_window_records_every_day(monkeypatch) -> None:
    _stub_get(monkeypatch, tone={}, counts={})
    end = date(2026, 8, 30)
    start = end - timedelta(days=89)

    assert gdelt.collect_symbol(ENTRY, start, end, keep_raw=False) == 90
    assert len(gdelt.read_store("AAPL")["days"]) == 90


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_a_non_json_reply_is_reported_with_the_body(monkeypatch) -> None:
    """GDELT answers a malformed query with HTTP 200 and plain text."""

    class FakeResponse:
        def read(self):
            return b"Your query was too broad."

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    with pytest.raises(gdelt.GdeltError, match="too broad"):
        gdelt._get({"mode": "timelinetone"})


def test_network_failure_exits_cleanly_rather_than_traceback(monkeypatch, capsys) -> None:
    """This runs from cron, where a traceback is a mail nobody reads."""
    monkeypatch.setattr(
        gdelt, "load_universe", lambda only=None: [ENTRY]
    )

    def boom(params):
        raise gdelt.GdeltError("Network error for artlist: timed out")

    monkeypatch.setattr(gdelt, "_get", boom)

    assert gdelt.main(["--inspect", "AAPL"]) == 1
    assert "GDELT unavailable" in capsys.readouterr().err


def test_ssl_context_prefers_certifi() -> None:
    """python.org macOS builds have no usable CA store without this."""
    import ssl

    assert isinstance(gdelt._ssl_context(), ssl.SSLContext)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def test_status_distinguishes_no_history_from_being_behind(monkeypatch, capsys) -> None:
    monkeypatch.setattr(gdelt, "load_universe", lambda only=None: [ENTRY])

    assert gdelt.main(["--status"]) == 0
    assert "No history collected yet" in capsys.readouterr().out

    _stub_get(monkeypatch, tone={}, counts={})
    yesterday = date.today() - timedelta(days=1)
    gdelt.collect_symbol(ENTRY, yesterday, yesterday, keep_raw=False)

    assert gdelt.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "No history collected yet" not in out
    assert "No gaps." in out


def test_status_counts_gaps_in_the_middle_of_a_range(monkeypatch, capsys) -> None:
    """A gap is a day that is gone for good once the window rolls past it."""
    monkeypatch.setattr(gdelt, "load_universe", lambda only=None: [ENTRY])
    _stub_get(monkeypatch, tone={}, counts={})

    gdelt.collect_symbol(ENTRY, date(2026, 8, 1), date(2026, 8, 2), keep_raw=False)
    gdelt.collect_symbol(ENTRY, date(2026, 8, 10), date(2026, 8, 11), keep_raw=False)

    gdelt.main(["--status"])
    out = capsys.readouterr().out
    assert "7 missing day(s)" in out


# --------------------------------------------------------------------------- #
# Rate limiting and retry
#
# GDELT states its limit only in the BODY of a 429, and sends no Retry-After.
# Both facts were confirmed against the live API, and getting either wrong is
# what produced "GDELT unavailable: HTTP 429 ... Too Many Requests" with no hint
# that the fix was simply to slow down.
# --------------------------------------------------------------------------- #


class _FakeHTTPError(Exception):
    """Stands in for urllib.error.HTTPError, which is file-like."""

    def __init__(self, code: int, body: bytes, reason: str = "error"):
        self.code = code
        self.reason = reason
        self._body = body

    def read(self) -> bytes:
        return self._body


@pytest.fixture
def no_waiting(monkeypatch):
    """Record sleeps instead of taking them. Keeps the suite fast."""
    slept: list[float] = []
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(gdelt, "_last_request_at", None)
    return slept


THROTTLE_BODY = (
    b"Please limit requests to one every 5 seconds or contact "
    b"kalev.leetaru5@gmail.com for larger queries."
)


def test_gdelts_own_words_reach_the_user(monkeypatch, no_waiting) -> None:
    """The regression that started this.

    The limit is stated ONLY in the 429 body. Formatting exc.reason and
    discarding exc.read() turned actionable guidance into a generic status line.
    """
    monkeypatch.setattr(gdelt.urllib.error, "HTTPError", _FakeHTTPError)

    def always_throttled(*a, **k):
        raise _FakeHTTPError(429, THROTTLE_BODY, "Too Many Requests")

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", always_throttled)

    with pytest.raises(gdelt.GdeltError) as caught:
        gdelt._get({"mode": "artlist"})

    message = str(caught.value)
    assert "one every 5 seconds" in message, "the body must survive into the error"
    assert "429" in message


def test_a_429_is_retried_and_can_succeed(monkeypatch, no_waiting) -> None:
    monkeypatch.setattr(gdelt.urllib.error, "HTTPError", _FakeHTTPError)
    calls = {"n": 0}

    class Ok:
        def read(self):
            return b'{"timeline": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _FakeHTTPError(429, THROTTLE_BODY, "Too Many Requests")
        return Ok()

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", flaky)

    assert gdelt._get({"mode": "timelinetone"}) == {"timeline": []}
    assert calls["n"] == 3
    # Backoff grows rather than hammering a throttle that is already angry.
    waits = [s for s in no_waiting if s >= 1]
    assert waits == [gdelt.RETRY_BACKOFF_S[0], gdelt.RETRY_BACKOFF_S[1]]


def test_backoff_gives_up_after_the_schedule(monkeypatch, no_waiting) -> None:
    monkeypatch.setattr(gdelt.urllib.error, "HTTPError", _FakeHTTPError)
    calls = {"n": 0}

    def always(*a, **k):
        calls["n"] += 1
        raise _FakeHTTPError(429, THROTTLE_BODY, "Too Many Requests")

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", always)

    with pytest.raises(gdelt.GdeltError):
        gdelt._get({"mode": "artlist"})

    assert calls["n"] == len(gdelt.RETRY_BACKOFF_S) + 1


def test_a_client_error_is_not_retried(monkeypatch, no_waiting) -> None:
    """404 and 400 are our bug. Retrying just hides it behind a delay."""
    monkeypatch.setattr(gdelt.urllib.error, "HTTPError", _FakeHTTPError)
    calls = {"n": 0}

    def not_found(*a, **k):
        calls["n"] += 1
        raise _FakeHTTPError(404, b"no such endpoint", "Not Found")

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", not_found)

    with pytest.raises(gdelt.GdeltError, match="404"):
        gdelt._get({"mode": "artlist"})
    assert calls["n"] == 1, "a 404 must fail immediately"


def test_requests_are_never_closer_than_the_interval(monkeypatch) -> None:
    """The clock gate, which is what actually keeps us under the limit.

    Asserted on the sleeps requested rather than on elapsed time, so the test
    stays fast and deterministic.
    """
    clock = {"t": 1000.0}
    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(gdelt.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(gdelt.time, "sleep", fake_sleep)
    monkeypatch.setattr(gdelt, "_last_request_at", None)
    monkeypatch.setattr(gdelt, "REQUEST_INTERVAL_S", 5.0)

    class Ok:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    # Each request "takes" 1.2s of wall clock, so a naive trailing sleep would
    # leave only 5s between starts but the gate must still see 5s of gap.
    def urlopen(*a, **k):
        clock["t"] += 1.2
        return Ok()

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", urlopen)

    for _ in range(3):
        gdelt._get({"mode": "timelinetone"})

    # First request waits for nobody; the next two each wait out the remainder.
    assert len(slept) == 2
    for wait in slept:
        assert wait == pytest.approx(5.0 - 1.2, abs=0.01)


def test_interval_is_configurable_from_the_cli(monkeypatch) -> None:
    """So a backfill can be slowed further without editing the source."""
    monkeypatch.setattr(gdelt, "load_universe", lambda only=None: [])
    original = gdelt.REQUEST_INTERVAL_S
    try:
        gdelt.main(["--status", "--interval", "9"])
        assert gdelt.REQUEST_INTERVAL_S == 9.0
    finally:
        gdelt.REQUEST_INTERVAL_S = original


def test_a_sustained_block_says_to_stop_rather_than_retry(monkeypatch, no_waiting) -> None:
    """Measured behaviour: an IP in the penalty box stays 429 well past 150s.

    When that happens the useful advice is the opposite of "try again" -- more
    requests from a throttled IP plausibly extend the block.
    """
    monkeypatch.setattr(gdelt.urllib.error, "HTTPError", _FakeHTTPError)
    monkeypatch.setattr(
        gdelt.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            _FakeHTTPError(429, THROTTLE_BODY, "Too Many Requests")
        ),
    )

    with pytest.raises(gdelt.GdeltError) as caught:
        gdelt._get({"mode": "artlist"})

    message = str(caught.value)
    assert "SUSTAINED" in message
    assert "Wait ~15 minutes" in message
    assert "may extend it" in message


def test_backoff_is_short_on_purpose() -> None:
    """Guard the reasoning above against a well-meaning 'make it more robust'.

    Lengthening this schedule makes the sustained-block case worse, not better.
    """
    assert len(gdelt.RETRY_BACKOFF_S) == 2
    assert sum(gdelt.RETRY_BACKOFF_S) <= 60, (
        "Long backoff against a per-IP penalty box is counterproductive -- see "
        "the comment on RETRY_BACKOFF_S before raising this."
    )


# --------------------------------------------------------------------------- #
# The false-zero bug
#
# Found by running the collector live for the first time. GDELT answered HTTP
# 200 with an EMPTY BODY; the collector parsed that into "no data for any day"
# and wrote articles=0, observed=True across the whole window. Those days then
# look collected -- no gap in --status, never re-fetched -- and ~3 months later
# the real numbers are gone. A false zero is strictly worse than a hole.
# --------------------------------------------------------------------------- #


def test_an_empty_200_is_not_recorded_as_zero_coverage(monkeypatch) -> None:
    """The regression. Must raise, and must write nothing at all."""
    monkeypatch.setattr(gdelt, "_get", lambda params: {})

    with pytest.raises(gdelt.GdeltError, match="no timeline"):
        gdelt.collect_symbol(ENTRY, date(2026, 10, 3), date(2026, 10, 5), keep_raw=False)

    assert not gdelt.store_path("AAPL").exists(), (
        "a failed fetch must leave no trace -- a partially written store is "
        "indistinguishable from a successful one"
    )


def test_a_real_timeline_with_a_quiet_day_still_records_zero(monkeypatch) -> None:
    """The legitimate case, which must keep working.

    GDELT answered properly and simply had nothing for the middle day. That IS
    observed zero coverage and should be recorded as such -- the point of the
    fix is to separate this from silence, not to stop recording zeros.
    """
    _stub_get(
        monkeypatch,
        tone={"2026-10-03": 1.5, "2026-10-05": -0.5},
        counts={"2026-10-03": 12, "2026-10-05": 7},
    )
    gdelt.collect_symbol(ENTRY, date(2026, 10, 3), date(2026, 10, 5), keep_raw=False)

    days = gdelt.read_store("AAPL")["days"]
    assert days["2026-10-03"]["articles"] == 12
    assert days["2026-10-04"] == {
        **days["2026-10-04"],
        "articles": 0,
        "tone": None,
        "observed": True,
    }
    assert days["2026-10-05"]["articles"] == 7


def test_has_timeline_discriminates() -> None:
    assert gdelt._has_timeline({"timeline": [{"series": "Article Count", "data": []}]})
    assert not gdelt._has_timeline({})
    assert not gdelt._has_timeline({"timeline": []})
    assert not gdelt._has_timeline(None)
    assert not gdelt._has_timeline("Please limit requests to one every 5 seconds")
