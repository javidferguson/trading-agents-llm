"""``PortfolioIntent`` -- ``portfolio-intent.yaml``, typed. Architecture §8.

The file has been read as a raw dict since Stage 3, which was fine while only
``render_intent`` touched it. Stage 6 makes it load-bearing in a way a dict is
not safe for: ``risk.max_position_pct`` is now the number that decides whether
an order is placed, and ``intent.get("risk", {}).get("max_position_pct")``
returning ``None`` would compare as "no limit" rather than failing.

So every limit is required, bounded, and validated at load. A missing cap is a
``ValidationError`` at startup, not a silently absent veto at order time.

**The four consumption channels keep this object in two halves (§8):**

* ``objective``, ``themes[].thesis`` and ``constraints`` are *prose for a
  model* -- channel 3, and they reach the Trader and Fund Manager only.
* everything under ``risk:`` is *a number for Python* -- channel 4, enforced by
  ``intent/compliance.py`` after sizing, and never presented to a model as
  negotiable.

``render_intent`` is the only path by which any of this reaches a prompt, and
``tests/test_intent_blindness.py`` is what keeps it that way.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Objective(BaseModel):
    """What the book is for. Prose, read by the trader and fund manager only."""

    horizon_days: int = Field(ge=1, le=750)
    benchmark: str = "SPY"
    description: str = ""

    @field_validator("benchmark")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


class RiskLimits(BaseModel):
    """Channel 4: **the hard veto.** Every field here is enforced in Python.

    None of these is a suggestion to a model. ``render_intent`` shows a few of
    them to the trader so it can propose something plausible, but the
    enforcement is ``compliance.py`` and it runs *after* the fund manager --
    "any violation blocks the order regardless of what the fund manager
    decided" (§9).

    Required rather than defaulted, with one exception noted below. A default
    would mean a typo'd key name silently installs a limit nobody chose.
    """

    per_trade_risk_pct: float = Field(gt=0, le=100)
    default_stop_pct: float = Field(gt=0, lt=100)
    max_position_pct: float = Field(gt=0, le=100)
    max_sector_pct: float = Field(gt=0, le=100)
    min_cash_pct: float = Field(ge=0, lt=100)
    max_gross_exposure_pct: float = Field(gt=0, le=200)
    max_positions: int = Field(ge=1)
    min_trade_usd: float = Field(ge=0)
    max_order_shares: int = Field(ge=1)
    max_adv_participation_pct: float = Field(gt=0, le=100)
    earnings_blackout_days_before: int = Field(ge=0)
    earnings_blackout_days_after: int = Field(ge=0)

    @model_validator(mode="after")
    def _limits_are_mutually_possible(self) -> "RiskLimits":
        """Catch a config that can never be satisfied, at load rather than at
        order time.

        Only one such pair exists. A risk cap above the position cap is
        decoration: ``cap_risk`` in §9's minimum-of-four could never be the
        binding term, so the limit would read as enforced and never enforce
        anything.

        **The pair that looks like a contradiction and is not** is
        ``min_cash_pct`` and ``max_gross_exposure_pct``, which sum to 105 in
        the shipped config. In a long-only unlevered book
        ``cash_pct + gross_pct == 100`` identically, so a 10% cash floor already
        caps gross at 90 and the 95% limit can never bind. That is redundancy,
        not impossibility -- no order is blocked by both, and the 95% figure
        starts mattering the moment ``universe.allow_shorts`` becomes true,
        since gross counts shorts twice. ``effective_gross_cap_pct`` below
        resolves which one actually binds, so compliance can name it.
        """
        if self.per_trade_risk_pct > self.max_position_pct:
            raise ValueError(
                f"per_trade_risk_pct ({self.per_trade_risk_pct}) exceeds "
                f"max_position_pct ({self.max_position_pct}). The risk cap would "
                "never be the binding one, which makes it decoration."
            )
        return self

    def effective_gross_cap_pct(self, *, allow_shorts: bool) -> float:
        """The gross-exposure ceiling that actually binds.

        Long-only: the cash floor implies a gross ceiling of
        ``100 - min_cash_pct``, and whichever of that and
        ``max_gross_exposure_pct`` is lower is the real limit. With shorts
        permitted the identity breaks -- gross counts both legs -- so the
        declared limit stands on its own.
        """
        if allow_shorts:
            return self.max_gross_exposure_pct
        return min(self.max_gross_exposure_pct, 100.0 - self.min_cash_pct)

    @property
    def stop_distance_pct(self) -> float:
        """The stop distance sizing divides by. Named so the formula reads as §9."""
        return self.default_stop_pct


class Theme(BaseModel):
    """A sleeve of the book. ``target_weight_pct`` is the **theme total**.

    That distinction has already cost one round of bad decisions: reading the
    Stage 3 output, the trader proposed 35% of equity in AVGO against a 10% cap
    because it read the AI theme's 35% total as AVGO's own target. Four of ten
    proposals exceeded the cap that way, which is why ``render_intent`` spells
    out the implied per-symbol figure and
    ``tests/test_intent_blindness.py`` asserts it still does.
    """

    name: str
    target_weight_pct: float = Field(ge=0, le=100)
    #: 0..1, and **prose only -- it is not a size multiplier.**
    #:
    #: This comment used to claim it "multiplies the target in sizing's
    #: ``conviction_w``", which was never true: nothing has ever read this
    #: field numerically. It reaches the trader and the fund manager through
    #: ``render_intent`` (channel 3), next to the theme's thesis, and its job
    #: is to say how much the desk believes that thesis when the model is
    #: choosing which gap to close.
    #:
    #: It is deliberately not wired into sizing, for a reason that is easy to
    #: miss: as a multiplier it would be **redundant with
    #: ``target_weight_pct``**. If the desk wants 28% in AI infrastructure, the
    #: target should read 28% -- writing 35% and relying on a 0.8 conviction to
    #: scale it down expresses the same intention less legibly, and leaves two
    #: numbers that can disagree about what the desk actually wants.
    conviction: float = Field(ge=0, le=1)
    thesis: str = ""
    exemplars: list[str] = Field(default_factory=list)

    @field_validator("exemplars")
    @classmethod
    def _upper(cls, value: list[str]) -> list[str]:
        return [s.strip().upper() for s in value]

    @property
    def implied_symbol_weight_pct(self) -> float:
        """The theme total spread evenly across its exemplars.

        What a per-symbol target would be if the theme were held equal-weight.
        Computed here rather than in a prompt because dividing is arithmetic
        and §2 says models do not do arithmetic.
        """
        count = len(self.exemplars) or 1
        return self.target_weight_pct / count

    def holds(self, symbol: str) -> bool:
        return symbol.strip().upper() in self.exemplars


class Universe(BaseModel):
    """What may be traded. Distinct from ``universe.yaml``, which is what is
    *collected* -- the second is deliberately wider (§13)."""

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    allow_shorts: bool = False

    @field_validator("include", "exclude")
    @classmethod
    def _upper(cls, value: list[str]) -> list[str]:
        return [s.strip().upper() for s in value]

    @property
    def tradeable(self) -> list[str]:
        """``include`` minus ``exclude``, in file order. Channel 1's first step."""
        excluded = set(self.exclude)
        return [s for s in self.include if s not in excluded]

    def admits(self, symbol: str) -> bool:
        return symbol.strip().upper() in set(self.tradeable)


class Cadence(BaseModel):
    """How often, and for how long a proposal stays executable."""

    max_decisions_per_day: int = Field(default=3, ge=1)
    proposal_ttl_hours: float = Field(default=18.0, gt=0)


class Execution(BaseModel):
    """The human gate. **There is no off switch (§9).**

    ``config.load_config`` already refuses a file that sets
    ``require_confirmation: false``. This validator is the second lock, so that
    a ``PortfolioIntent`` built in code -- a test fixture, a replay harness --
    cannot construct one either. The gate was inverted once in the ORB engine's
    options scanner: setting it false still prompted, then returned True
    regardless of the answer. Two locks is the right number for that.
    """

    require_confirmation: Literal[True] = True
    confirm_by: Literal["ticker", "symbol"] = "ticker"

    @model_validator(mode="after")
    def _gate_is_intact(self) -> "Execution":
        if self.require_confirmation is not True:
            raise ValueError(
                "the human confirmation gate has no off switch -- every order "
                "on this path requires explicit approval (§9)"
            )
        return self


class PortfolioIntent(BaseModel):
    """The whole file. Built by ``load_intent()`` in ``intent/engine.py``."""

    objective: Objective
    risk: RiskLimits
    universe: Universe
    themes: list[Theme] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    cadence: Cadence = Field(default_factory=Cadence)
    execution: Execution = Field(default_factory=Execution)

    @model_validator(mode="after")
    def _themes_are_coherent(self) -> "PortfolioIntent":
        """Two checks that would otherwise surface as a puzzling drift table.

        Theme targets summing above ``max_gross_exposure_pct`` is not
        necessarily wrong -- the gaps simply never all close -- but it means
        the drift table permanently asks for more risk than the book may hold,
        and that is worth knowing at load rather than inferring from a trader
        that keeps proposing buys which compliance keeps blocking.
        """
        total = sum(t.target_weight_pct for t in self.themes)
        if total > self.risk.max_gross_exposure_pct:
            raise ValueError(
                f"theme targets sum to {total}% but max_gross_exposure_pct is "
                f"{self.risk.max_gross_exposure_pct}%. The drift table would "
                "permanently request more exposure than compliance allows."
            )

        # An exemplar outside the tradeable universe produces a theme gap that
        # can never be closed: the drift table asks for it, the pre-filter
        # refuses to run it, and nothing says why.
        tradeable = set(self.universe.tradeable)
        for theme in self.themes:
            orphans = sorted(set(theme.exemplars) - tradeable)
            if orphans:
                raise ValueError(
                    f"theme {theme.name!r} names exemplars {orphans} that are "
                    "not in universe.include (or are excluded). Their gap could "
                    "never be closed."
                )
        return self

    # --- lookups the engine and compliance need -----------------------------

    def theme_for(self, symbol: str) -> Theme | None:
        """The first theme claiming ``symbol``, or ``None``.

        First rather than all: a symbol in two themes would be double-counted
        in the drift table, and ``_themes_do_not_overlap`` below rejects that
        at load, so "first" is always "only".
        """
        key = symbol.strip().upper()
        return next((t for t in self.themes if t.holds(key)), None)

    @model_validator(mode="after")
    def _themes_do_not_overlap(self) -> "PortfolioIntent":
        seen: dict[str, str] = {}
        for theme in self.themes:
            for symbol in theme.exemplars:
                if symbol in seen:
                    raise ValueError(
                        f"{symbol} appears in both {seen[symbol]!r} and "
                        f"{theme.name!r}. Its weight would count toward two "
                        "theme targets at once."
                    )
                seen[symbol] = theme.name
        return self

    def theme_named(self, name: str) -> Theme | None:
        return next((t for t in self.themes if t.name == name), None)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "PortfolioIntent":
        return cls.model_validate(body)


__all__ = [
    "Cadence",
    "Execution",
    "Objective",
    "PortfolioIntent",
    "RiskLimits",
    "Theme",
    "Universe",
]
