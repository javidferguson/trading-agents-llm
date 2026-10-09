"""``scripts/seed_portfolio.py`` -- the two commands that maintain the book.

The user never hand-edits ``data/portfolio.yaml``; these two commands do, and
neither may touch IB. The property worth a test is the one that is easy to get
backwards: **a refresh must not reset the staleness clock.**
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from research_desk.models.portfolio import PortfolioSnapshot, Position

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def seeder():
    spec = importlib.util.spec_from_file_location(
        "seed_portfolio", REPO_ROOT / "scripts" / "seed_portfolio.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_portfolio"] = module
    spec.loader.exec_module(module)
    return module


def book(*marks: tuple[str, date]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=min(d for _, d in marks),
        cash=1_000.0,
        positions=[
            Position(symbol=s, quantity=10, avg_cost=100.0, last_price=100.0,
                     marked_on=d)
            for s, d in marks
        ],
    )


def test_as_of_is_the_oldest_mark_not_the_newest(seeder) -> None:
    """**The staleness clock cannot be reset by a refresh.**

    If ``as_of`` took the newest mark, re-running a refresh against a bars
    cache that has not been updated would move the date forward and declare a
    three-week-old book fresh. Taking the oldest means the book is only as
    fresh as its stalest position, which is the honest reading.
    """
    stale = book(("AAA", date(2026, 10, 6)), ("BBB", date(2026, 9, 1)))
    assert stale.as_of == date(2026, 9, 1)
    assert stale.is_stale(date(2026, 10, 6))


def test_a_refresh_keeps_a_position_whose_bars_are_missing(seeder, monkeypatch) -> None:
    """Dropping it would silently shrink the book, which changes every weight."""
    monkeypatch.setattr(seeder, "_cached_bars", lambda: {
        "AAA": [{"day": "2026-10-06", "close": 120.0}],
    })
    before = book(("AAA", date(2026, 10, 1)), ("BBB", date(2026, 10, 1)))
    after = seeder.refresh(before)

    assert {p.symbol for p in after.positions} == {"AAA", "BBB"}
    assert after.get("AAA").last_price == pytest.approx(120.0)
    # BBB keeps its old mark, and therefore holds the book's as_of back.
    assert after.get("BBB").last_price == pytest.approx(100.0)
    assert after.as_of == date(2026, 10, 1)


def test_a_refresh_marks_the_source_as_cache(seeder, monkeypatch) -> None:
    """``source`` records which path produced the numbers, for the same reason
    ``Bar.source`` does: when two runs disagree you want to know which book
    each one saw."""
    monkeypatch.setattr(seeder, "_cached_bars", lambda: {
        "AAA": [{"day": "2026-10-06", "close": 120.0}],
    })
    after = seeder.refresh(book(("AAA", date(2026, 10, 1))))
    assert after.source == "cache"


def test_the_seed_refuses_rather_than_inventing_a_price(seeder, monkeypatch) -> None:
    monkeypatch.setattr(seeder, "_cached_bars", lambda: {"NVDA": [
        {"day": "2026-10-06", "close": 240.0}
    ]})
    with pytest.raises(SystemExit, match="will not invent one"):
        seeder.seed()


def test_the_seed_book_is_at_max_positions_by_design(seeder) -> None:
    """The gate's drift table is built around a full book: it is what makes
    the AI-infrastructure gap uncloseable and the decision interesting."""
    from research_desk.intent.engine import load_intent

    assert len(seeder.SEED_BOOK) == load_intent().risk.max_positions


def test_the_seed_prices_only_real_cached_closes(seeded_book, seeder) -> None:
    """Every number the SEED produces traces to a bar on disk, so the weights
    are arithmetic over real data rather than figures chosen to look tidy.

    **Reads the seed, not ``load_portfolio()``, and that is the whole fix.**
    This asserted the property of "the committed book" while loading
    ``data/portfolio.yaml``, which stopped being the committed book at Stage 7:
    ``execute --sync-book`` now writes it from the broker with ``source: ib``
    and marks at **average cost**, which by design do not match any cached
    close. The first real trading session broke it -- AMD at an average cost of
    645.86 against a newest close of 613.87 -- and the failure message said
    "re-run `make portfolio-refresh`", which is advice from before the broker
    owned that file.

    Fourth time this project has been bitten by a test reading state instead of
    a fixture. The seed recipe is the fixture; the file is state.
    """
    cache = seeder._cached_bars()
    for held in seeded_book.positions:
        bars = cache.get(held.symbol)
        assert bars, f"{held.symbol} is in the seed with no cached bars"
        assert held.last_price == pytest.approx(float(bars[-1]["close"])), (
            f"{held.symbol}'s seeded mark does not match the newest cached "
            "close, so the seed is not pricing from the cache"
        )


def test_the_seeder_reads_the_cache_and_never_constructs_a_provider(seeder) -> None:
    """It must run with the Gateway down. Asserted on the source, because the
    temptation is one convenience import away."""
    source = (REPO_ROOT / "scripts" / "seed_portfolio.py").read_text()
    for banned in ("ib_async", "ProviderRegistry", "IBProvider", "httpx"):
        assert banned not in source, (
            f"seed_portfolio.py mentions {banned}. It reads the bars cache and "
            "must not be able to fetch."
        )
