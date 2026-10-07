"""``PortfolioSnapshot`` -- what the book holds, as a file rather than a broker.

Stage 6 sizing needs two numbers the rest of the design does not have: how much
equity there is, and how much of it is already in the symbol under
consideration. The exit gate requires **zero IB contact**, so this comes from
``config/portfolio.yaml`` and never from a socket.

Three rules, and each one exists because the alternative is a silently wrong
position size.

**Equity is derived, never stored.** ``equity = cash + sum(market_value)``, the
way IB reports NetLiquidation. A stored equity figure and a position list can
disagree, and the failure is invisible: sizing against an equity that is 20%
too high overstates every target weight by 20% and nothing raises.

**Marks are stored per position, not fetched.** ``decide`` builds a
``MarketSnapshot`` for *one* symbol -- the one being decided -- so it has no
price for the other seven holdings and cannot mark the book itself. Each
position therefore carries its own ``last_price`` and the day it came from.
``desk portfolio --refresh`` re-marks from the bars cache; at Stage 7 ``execute``
writes this file from IB directly. Neither path needs a live connection here.

**Staleness blocks rather than warns.** ``as_of`` is the date of the *marks*,
not the day the file was written -- so re-running a refresh against a stale bars
cache cannot reset the clock. A book that is too old to trust is a compliance
violation (``portfolio_stale``), because sizing against last week's weights is
exactly how a position gets doubled: the drift table says there is room, and
there is not. Failure direction is HOLD (§5).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

#: Default tolerance for stale marks, in calendar days. A long weekend plus a
#: public holiday is four; seven leaves one slack day and no more.
DEFAULT_STALE_AFTER_DAYS = 7


class Position(BaseModel):
    """One holding, marked to a stored price.

    ``quantity`` is signed -- negative is short -- even though
    ``universe.allow_shorts`` is false today. The sign convention costs nothing
    now and means ``market_value`` does not have to be reinterpreted later;
    §9's sizing explicitly "assumes long-only", so the *sizing* path rejects a
    short rather than this model pretending shorts are impossible.
    """

    symbol: str
    quantity: float
    #: Average cost per share. Used for P&L reporting only -- **never for
    #: sizing.** Sizing works on market value, because what the book risks is
    #: what the position is worth now, not what it cost.
    avg_cost: float = Field(ge=0)
    last_price: float = Field(gt=0)
    #: The day ``last_price`` came from. Per position, because a refresh can
    #: legitimately find one symbol's bars fresher than another's.
    marked_on: date

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _not_a_zero_position(self) -> "Position":
        """A zero-quantity row is a closed position, not a holding.

        Left in the file it would count toward ``max_positions`` and occupy a
        slot that is not actually occupied -- which blocks a trade for a reason
        that is not true.
        """
        if self.quantity == 0:
            raise ValueError(
                f"{self.symbol} has quantity 0. A closed position is removed "
                "from the file, not kept at zero -- it would otherwise count "
                "toward max_positions and block a trade for no reason."
            )
        return self

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def unrealized_usd(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pct(self) -> float | None:
        basis = abs(self.cost_basis)
        return None if basis == 0 else 100.0 * self.unrealized_usd / basis

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


class PortfolioSnapshot(BaseModel):
    """The book, as of a date. The second input to sizing after ``PortfolioIntent``."""

    #: The date of the MARKS -- the oldest ``marked_on`` across positions, not
    #: the day the file was saved. See the module docstring.
    as_of: date
    cash: float
    base_currency: str = "USD"
    positions: list[Position] = Field(default_factory=list)

    #: Where the numbers came from. ``seed`` is a hand-built book for the Stage
    #: 6 gate, ``cache`` is re-marked from the bars cache, ``ib`` is written by
    #: ``execute`` at Stage 7. Recorded for the same reason ``Bar.source`` is:
    #: when two runs disagree you will want to know which book each saw.
    source: Literal["seed", "cache", "ib"] = "seed"
    stale_after_days: int = Field(default=DEFAULT_STALE_AFTER_DAYS, ge=1)
    note: str = ""

    @model_validator(mode="after")
    def _one_row_per_symbol(self) -> "PortfolioSnapshot":
        """Two rows for one symbol would make ``position_value`` ambiguous.

        IB reports one aggregated position per contract, so two rows means the
        file was edited by hand and the halves disagree. Sizing would see
        whichever ``next()`` found first.
        """
        seen = [p.symbol for p in self.positions]
        duplicates = sorted({s for s in seen if seen.count(s) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate positions for {duplicates}. One row per symbol -- "
                "aggregate them, the way IB does."
            )
        return self

    # --- derived, so nothing can disagree with anything else ----------------

    @property
    def long_value(self) -> float:
        return sum(p.market_value for p in self.positions if not p.is_short)

    @property
    def gross_value(self) -> float:
        """Absolute exposure. Shorts add to gross rather than netting against it."""
        return sum(abs(p.market_value) for p in self.positions)

    @property
    def net_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def equity(self) -> float:
        """NetLiquidation: cash plus the net value of what is held."""
        return self.cash + self.net_value

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def symbols(self) -> set[str]:
        return {p.symbol for p in self.positions}

    def get(self, symbol: str) -> Position | None:
        key = symbol.strip().upper()
        return next((p for p in self.positions if p.symbol == key), None)

    def position_value(self, symbol: str) -> float:
        """§9's ``pf.position_value(symbol)``. Zero for an unheld symbol."""
        held = self.get(symbol)
        return 0.0 if held is None else held.market_value

    def _pct(self, value: float) -> float:
        equity = self.equity
        # A zero-or-negative equity book cannot be sized against at all, and
        # returning 0.0 here would read as "no exposure" -- the most dangerous
        # possible answer. compliance.py rejects the book outright; this just
        # refuses to invent a percentage.
        return 0.0 if equity <= 0 else 100.0 * value / equity

    def weight_pct(self, symbol: str) -> float:
        return self._pct(self.position_value(symbol))

    @property
    def cash_pct(self) -> float:
        return self._pct(self.cash)

    @property
    def gross_exposure_pct(self) -> float:
        return self._pct(self.gross_value)

    # --- staleness ----------------------------------------------------------

    def staleness_days(self, as_of: date) -> int:
        """Calendar days between the marks and the decision date.

        Negative when the marks are *ahead* of the decision date, which happens
        on a replay run against an older ``as_of`` and is a look-ahead
        violation rather than freshness (§7.4). ``is_stale`` treats it as such.
        """
        return (as_of - self.as_of).days

    def is_stale(self, as_of: date) -> bool:
        drift = self.staleness_days(as_of)
        return drift < 0 or drift > self.stale_after_days

    def staleness_reason(self, as_of: date) -> str | None:
        """Why the book cannot be trusted on ``as_of``, or ``None``."""
        drift = self.staleness_days(as_of)
        if drift < 0:
            return (
                f"the book is marked {self.as_of} but the decision date is "
                f"{as_of} -- the marks are {-drift} day(s) in the FUTURE, which "
                "is look-ahead bias, not freshness (§7.4)"
            )
        if drift > self.stale_after_days:
            return (
                f"the book is marked {self.as_of}, {drift} days before {as_of} "
                f"(limit {self.stale_after_days}). Re-mark it with "
                "`desk portfolio --refresh` before sizing against it"
            )
        return None

    # --- loading ------------------------------------------------------------

    @classmethod
    def flat(cls, equity: float, *, as_of: date, note: str = "") -> "PortfolioSnapshot":
        """An all-cash book. The honest starting state, and what tests use."""
        return cls(as_of=as_of, cash=equity, positions=[], source="seed", note=note)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "PortfolioSnapshot":
        return cls.model_validate(body)

    @classmethod
    def load(cls, path: Path) -> "PortfolioSnapshot":
        """Read ``config/portfolio.yaml``.

        Raises ``FileNotFoundError`` rather than defaulting to an empty book: a
        missing file sized as all-cash would propose a full-size opening trade
        in a symbol that may already be at its cap.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Sizing needs a book; an absent file is "
                "not an empty one. Generate a seed with `desk portfolio --seed`."
            )
        body = yaml.safe_load(path.read_text()) or {}
        if not isinstance(body, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
        return cls.from_dict(body)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Round-trippable plain data, for writing the file back out."""
        return {
            "as_of": self.as_of.isoformat(),
            "source": self.source,
            "base_currency": self.base_currency,
            "stale_after_days": self.stale_after_days,
            "note": self.note,
            "cash": round(self.cash, 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 4),
                    "last_price": round(p.last_price, 4),
                    "marked_on": p.marked_on.isoformat(),
                }
                for p in self.positions
            ],
        }


__all__ = ["DEFAULT_STALE_AFTER_DAYS", "PortfolioSnapshot", "Position"]
