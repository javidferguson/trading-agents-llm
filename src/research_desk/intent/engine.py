"""``compute_gaps()`` and the pre-filter. Channels 1 and 2, both **no LLM** (§8).

> *"``intent_engine.compute_gaps()`` -> per symbol and theme: current weight,
> target, band, gap in % and USD. The Trader node gets this as a table. **Its
> job is choosing which gap to close, not inventing allocations.** This is the
> biggest reliability win -- it converts an open-ended "how much should we buy"
> into a bounded "close this gap or don't", which small models handle far
> better."*

The claim is specific enough to check, and FOLLOWUPS.md recorded the failure it
is supposed to fix before this stage started: reading the Stage 3 output, qwen3
*"still infers a current position occasionally ('the position is small'),
despite the prompt stating that portfolio state is unknown."* It was inferring
because it had nothing. Now it has a table.

Three things this module deliberately does **not** do:

* **It does not decide.** Every number here is arithmetic over the book and the
  intent. No ranking, no "best gap", no recommendation -- that is the trader's
  job and giving it a pre-ranked list would make the trader a rubber stamp.
* **It does not clamp.** A gap larger than ``max_position_pct`` is reported at
  its true size. ``sizing.py`` applies the caps, and conflating the two would
  hide *why* an order came out smaller than the gap.
* **It does not touch a model.** No prompt, no router, no network beyond the
  earnings estimate the caller passes in.

**Bands.** §8 mentions an optional ``targets:`` block with bands;
``portfolio-intent.yaml`` has none, so the band is derived --
``max(BAND_FLOOR_PCT, target * BAND_FRACTION)`` -- and a gap inside it is
reported as ``in_band``. The band is what makes "or don't" a real option: with
no tolerance, every symbol is permanently off target by some rounding error and
the table reads as a to-do list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..config import CONFIG_DIR, REPO_ROOT, load_yaml
from ..models.intent import PortfolioIntent, Theme
from ..models.portfolio import PortfolioSnapshot

#: Tolerance around a target, as a fraction of the target itself. A 35% theme
#: gets +-8.75pp; a 3% symbol gets the floor below. Proportional because a
#: fixed band is either meaningless on a large target or impossible on a small
#: one.
BAND_FRACTION = 0.25

#: Minimum band in percentage points. Below this, a day's price move alone
#: pushes a symbol out of band and the table cries wolf.
#:
#: **1.0 -> 0.5 on 2026-10-07, and it was required rather than cosmetic.** The
#: floor was calibrated when the tradeable universe was 14 symbols and
#: per-symbol targets ran 3-8%. At 31 symbols the tail themes land on 1.00%
#: targets, and at a 1.0pp floor ``band_for(1.0) == 1.0`` -- so an *unheld*
#: symbol's gap of exactly 1.0pp is not GREATER than its band, reads ``in
#: band``, and is invisible to the trader forever. A whole theme can go dark
#: that way without anything erroring.
#:
#: 0.5 keeps what the floor is for -- one day's price move must not flip a
#: symbol out of band -- while leaving a 1% target actionable.
#: ``tests/intent/test_engine.py`` asserts every shipped theme's per-symbol
#: target exceeds its own band, so the next widening fails loudly here instead.
BAND_FLOOR_PCT = 0.5

#: The file the book lives in, and the directory matters.
#:
#: **``data/``, not ``config/``, and that is the second half of a lesson.** The
#: book was committed through Stage 6 and then gitignored, because it is state
#: rather than configuration. It kept living in ``config/`` anyway, and Stage 7
#: showed why that was wrong: the ``execute`` service mounts ``config`` as
#: READ-ONLY, deliberately, so the process that places orders can never rewrite
#: ``portfolio-intent.yaml`` and relax its own risk limits. Correct -- and it
#: also made it impossible for ``execute`` to write the book it is supposed to
#: own, which surfaced as `OSError: Read-only file system` the first time a
#: real fill needed recording.
#:
#: ``config/`` is policy that Python enforces and ``execute`` must not touch.
#: ``data/`` is state: gitignored, mounted read-write, and already home to the
#: journal and the proposals. The book belongs with those.
PORTFOLIO_FILE = "portfolio.yaml"

#: Where state lives. Separate from ``CONFIG_DIR`` on purpose -- see above.
DATA_DIR = REPO_ROOT / "data"


def load_intent(*, config_dir: Path | None = None) -> PortfolioIntent:
    """Read and validate ``portfolio-intent.yaml`` into the typed object."""
    return PortfolioIntent.from_dict(
        load_yaml("portfolio-intent.yaml", config_dir=config_dir)
    )


def portfolio_path(data_dir: Path | None = None) -> Path:
    """Where the book lives. ``data/``, not ``config/`` -- see PORTFOLIO_FILE."""
    return (data_dir or DATA_DIR) / PORTFOLIO_FILE


def load_portfolio(*, data_dir: Path | None = None) -> PortfolioSnapshot:
    """Read ``data/portfolio.yaml``.

    **Why this is not one of the four config files, and not in
    ``config_hash``.** It is *state*, not configuration: it changes every time
    the book is marked or a fill lands. Folding it into ``config_hash`` would
    change the hash daily and make the Stage 8 "are these two runs comparable"
    check answer no to every pair. What goes into ``DecisionState`` instead is
    the whole book, so a replay sees the weights the decision was made against.
    """
    return PortfolioSnapshot.load(portfolio_path(data_dir))


def sector_map(*, config_dir: Path | None = None) -> dict[str, str]:
    """``{symbol: sector_etf}`` from ``universe.yaml``.

    The sector classification the ``max_sector_pct`` veto groups by. It is a
    hand-maintained dict and ``universe.yaml`` says why: *"Free sector
    classification is poor, so a hand-maintained dict for a small universe is
    12 lines and honest."* It already exists for the relative-strength metric,
    so the compliance check gets it for nothing.

    A symbol with no ``sector_etf`` -- the benchmark itself -- is simply absent,
    and ``compliance.py`` treats an unknown sector as its own group rather than
    pooling every unclassified name into one bucket that trips the cap.
    """
    body = load_yaml("universe.yaml", config_dir=config_dir)
    out: dict[str, str] = {}
    for row in body.get("symbols") or []:
        symbol = str(row.get("symbol", "")).strip().upper()
        etf = row.get("sector_etf")
        if symbol and etf:
            out[symbol] = str(etf).strip().upper()
    return out


def band_for(target_pct: float) -> float:
    """Tolerance in percentage points around ``target_pct``."""
    return max(BAND_FLOOR_PCT, abs(target_pct) * BAND_FRACTION)


@dataclass(frozen=True)
class SymbolDrift:
    """One row of the per-symbol drift table."""

    symbol: str
    theme: str | None
    held: bool
    quantity: float
    current_value_usd: float
    current_weight_pct: float
    #: The theme's total spread across its exemplars, capped at
    #: ``max_position_pct``. Zero for a symbol in no theme: the book has no
    #: mandate for it, which is a fact the trader should see rather than a
    #: licence to size it freely.
    target_weight_pct: float
    band_pct: float
    #: Target minus current. Positive means underweight -- room to add.
    gap_pct: float
    gap_usd: float
    #: Why ``target_weight_pct`` is what it is, when it is not just the theme
    #: share -- e.g. capped by ``max_position_pct``.
    target_note: str = ""

    @property
    def in_band(self) -> bool:
        return abs(self.gap_pct) <= self.band_pct

    @property
    def direction(self) -> str:
        if self.in_band:
            return "in band"
        return "underweight" if self.gap_pct > 0 else "overweight"


@dataclass(frozen=True)
class ThemeDrift:
    """One row of the per-theme drift table. The target is the theme TOTAL."""

    name: str
    conviction: float
    target_weight_pct: float
    current_weight_pct: float
    band_pct: float
    gap_pct: float
    gap_usd: float
    held: tuple[str, ...] = ()
    unheld: tuple[str, ...] = ()

    @property
    def in_band(self) -> bool:
        return abs(self.gap_pct) <= self.band_pct

    @property
    def direction(self) -> str:
        if self.in_band:
            return "in band"
        return "underweight" if self.gap_pct > 0 else "overweight"


@dataclass(frozen=True)
class DriftTable:
    """Channel 2's output: the whole table, plus the book-level facts.

    The book-level fields are here rather than left for the trader to infer
    because every one of them is arithmetic (§2) and three of them --
    ``slots_free``, ``cash_pct``, ``addable_usd`` -- are the difference between
    a proposal that can be filled and one compliance will veto.
    """

    as_of: date
    equity: float
    cash_usd: float
    cash_pct: float
    gross_exposure_pct: float
    position_count: int
    max_positions: int
    #: Marks date of the book, so a stale table is visible in the table.
    book_as_of: date
    symbols: tuple[SymbolDrift, ...] = ()
    themes: tuple[ThemeDrift, ...] = ()
    unthemed_holdings: tuple[str, ...] = ()

    @property
    def slots_free(self) -> int:
        return max(self.max_positions - self.position_count, 0)

    def row(self, symbol: str) -> SymbolDrift | None:
        key = symbol.strip().upper()
        return next((r for r in self.symbols if r.symbol == key), None)

    def theme_row(self, name: str) -> ThemeDrift | None:
        return next((r for r in self.themes if r.name == name), None)


def _symbol_target(
    intent: PortfolioIntent, theme: Theme | None
) -> tuple[float, str]:
    """The per-symbol target weight, and why.

    A theme target is a *total* across its exemplars (§8), so the per-symbol
    figure is the theme share -- and then capped, because a two-exemplar 35%
    theme would otherwise imply 17.5% each against a 10% ceiling. That
    arithmetic is done here rather than in a prompt for the reason §2 gives,
    and because getting it wrong in a prompt is exactly what happened at
    Stage 3: four of ten proposals read the theme total as the symbol's target.
    """
    if theme is None:
        return 0.0, "in no theme -- the book has no mandate for this symbol"

    share = theme.implied_symbol_weight_pct
    cap = intent.risk.max_position_pct
    if share > cap:
        return cap, (
            f"theme share would be {share:.2f}% but max_position_pct caps a "
            f"single symbol at {cap:.2f}%"
        )
    return share, f"{theme.target_weight_pct:g}% theme total / {len(theme.exemplars)} exemplars"


def compute_gaps(
    intent: PortfolioIntent,
    portfolio: PortfolioSnapshot,
    *,
    as_of: date | None = None,
) -> DriftTable:
    """The drift table. Pure arithmetic over the book and the intent.

    Covers every tradeable symbol, not just the held ones -- an empty row for a
    themed symbol the book does not hold *is* the information ("this gap is
    fully open"), and omitting it would make the trader infer the universe from
    the holdings.
    """
    when = as_of or portfolio.as_of
    equity = portfolio.equity

    def usd(pct: float) -> float:
        return equity * pct / 100.0

    rows: list[SymbolDrift] = []
    for symbol in intent.universe.tradeable:
        theme = intent.theme_for(symbol)
        target, note = _symbol_target(intent, theme)
        held = portfolio.get(symbol)
        current_pct = portfolio.weight_pct(symbol)
        gap_pct = target - current_pct
        rows.append(SymbolDrift(
            symbol=symbol,
            theme=theme.name if theme else None,
            held=held is not None,
            quantity=held.quantity if held else 0.0,
            current_value_usd=portfolio.position_value(symbol),
            current_weight_pct=current_pct,
            target_weight_pct=target,
            band_pct=band_for(target),
            gap_pct=gap_pct,
            gap_usd=usd(gap_pct),
            target_note=note,
        ))

    theme_rows: list[ThemeDrift] = []
    for theme in intent.themes:
        current_pct = sum(portfolio.weight_pct(s) for s in theme.exemplars)
        gap_pct = theme.target_weight_pct - current_pct
        theme_rows.append(ThemeDrift(
            name=theme.name,
            conviction=theme.conviction,
            target_weight_pct=theme.target_weight_pct,
            current_weight_pct=current_pct,
            band_pct=band_for(theme.target_weight_pct),
            gap_pct=gap_pct,
            gap_usd=usd(gap_pct),
            held=tuple(s for s in theme.exemplars if portfolio.get(s) is not None),
            unheld=tuple(s for s in theme.exemplars if portfolio.get(s) is None),
        ))

    # A holding in no theme counts toward gross exposure and max_positions but
    # toward no theme target, so it is invisible in the theme table. Named
    # explicitly: an untracked position is how a book drifts without the drift
    # table noticing.
    themed = {s for t in intent.themes for s in t.exemplars}
    unthemed = tuple(sorted(portfolio.symbols - themed))

    return DriftTable(
        as_of=when,
        equity=equity,
        cash_usd=portfolio.cash,
        cash_pct=portfolio.cash_pct,
        gross_exposure_pct=portfolio.gross_exposure_pct,
        position_count=portfolio.position_count,
        max_positions=intent.risk.max_positions,
        book_as_of=portfolio.as_of,
        symbols=tuple(rows),
        themes=tuple(theme_rows),
        unthemed_holdings=unthemed,
    )


# --------------------------------------------------------------------------- #
# Channel 1 -- the deterministic pre-filter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidates:
    """Today's tradeable set, and why everything else was dropped.

    > *"Universe minus exclusions minus earnings blackout minus
    > at-max-positions -> today's candidates, computed before any model runs.
    > Saves money and removes a class of hallucination."* (§8)

    ``rejected`` carries a reason per symbol rather than a count, because "11
    of 14 symbols were filtered" is unactionable and "11 of 14 were filtered
    because the book is at max_positions" is a config review.
    """

    as_of: date
    eligible: tuple[str, ...] = ()
    rejected: dict[str, str] = field(default_factory=dict)

    def admits(self, symbol: str) -> bool:
        return symbol.strip().upper() in set(self.eligible)

    def reason(self, symbol: str) -> str | None:
        return self.rejected.get(symbol.strip().upper())


def candidates(
    intent: PortfolioIntent,
    portfolio: PortfolioSnapshot,
    *,
    as_of: date,
    blackouts: dict[str, str] | None = None,
) -> Candidates:
    """Channel 1. Runs before any model, and costs nothing.

    ``blackouts`` maps symbol -> reason and is computed by the caller via
    ``earnings.blackout_reason``, which is async because it reads EDGAR. Passed
    in rather than fetched here so this stays pure and synchronously testable.

    **At ``max_positions``, an already-held symbol is still a candidate.** It
    occupies a slot it already has, so trimming it, closing it, or adding to it
    changes no slot count. Filtering it out would make a full book untradeable
    rather than merely unable to open something new -- and the first thing you
    want to do with a full book is rebalance it.
    """
    blocked = {k.strip().upper(): v for k, v in (blackouts or {}).items()}
    excluded = set(intent.universe.exclude)
    held = portfolio.symbols
    at_capacity = portfolio.position_count >= intent.risk.max_positions

    eligible: list[str] = []
    rejected: dict[str, str] = {}

    for symbol in intent.universe.include:
        if symbol in excluded:
            rejected[symbol] = "excluded by universe.exclude"
            continue
        if symbol in blocked:
            rejected[symbol] = blocked[symbol]
            continue
        if at_capacity and symbol not in held:
            rejected[symbol] = (
                f"the book holds {portfolio.position_count} positions against a "
                f"max_positions of {intent.risk.max_positions}, so no new "
                "symbol can be opened -- held symbols remain tradeable"
            )
            continue
        eligible.append(symbol)

    return Candidates(as_of=as_of, eligible=tuple(eligible), rejected=rejected)


async def blackout_map(
    registry: Any,
    symbols: list[str],
    as_of: date,
    *,
    days_before: int,
    days_after: int,
) -> dict[str, str]:
    """Earnings blackout reasons for ``symbols``, for ``candidates()``.

    Only *usable* estimates filter a symbol out of the pre-filter. An unknown
    one does not: dropping every symbol whose earnings date could not be
    estimated would remove the FPIs and every ETF from the universe on the
    strength of missing data. ``compliance.py`` still records the unknown as a
    ``warn`` on whatever symbol is actually decided, which is the right place
    for it -- a human sees it there.
    """
    from .earnings import blackout_reason, estimate_next_report

    out: dict[str, str] = {}
    for symbol in symbols:
        estimate = await estimate_next_report(registry, symbol, as_of)
        if not estimate.is_usable:
            continue
        reason = blackout_reason(
            estimate, as_of, days_before=days_before, days_after=days_after
        )
        if reason:
            out[symbol] = reason
    return out


__all__ = [
    "BAND_FLOOR_PCT",
    "BAND_FRACTION",
    "Candidates",
    "DriftTable",
    "SymbolDrift",
    "ThemeDrift",
    "band_for",
    "blackout_map",
    "candidates",
    "compute_gaps",
    "load_intent",
    "DATA_DIR",
    "load_portfolio",
    "portfolio_path",
    "sector_map",
]
