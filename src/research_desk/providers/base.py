"""The provider contract.

Four rules, from architecture §2 and §7.4, all cheap now and expensive to
retrofit:

1. **Everything goes through ``registry.py``.** Nothing imports a provider
   module directly (``tests/test_layering.py`` enforces it). That is what makes
   these functions promotable to real tools later as a config change rather
   than a refactor.
2. **Tool-shaped signatures.** Fully typed parameters, no ``**kwargs``,
   JSON-serialisable returns, and a docstring written as if it were already
   the tool description a model would read -- so the schema can be *generated*
   from the signature rather than hand-written twice.
3. **``as_of`` on every method, always.** The point-in-time guard lives in the
   function, not the caller, because a tool a model can call at an arbitrary
   moment is exactly where look-ahead bias creeps back in.
4. **Every provider declares ``supports_point_in_time``.** In
   ``mode="replay"`` the cache is the only permitted source, and a provider
   that cannot do point-in-time **raises** rather than degrading quietly.
   Look-ahead bias is the number one way agentic backtests produce fictional
   Sharpe ratios (§15.1).
"""

from __future__ import annotations

from dataclasses import dataclass


class ProviderError(RuntimeError):
    """A provider could not serve a request."""


class PointInTimeError(ProviderError):
    """A provider was asked for something it cannot answer honestly for a past date.

    Raised rather than degraded, deliberately. A provider that returns
    *today's* value for a replay of 2024 produces a backtest that looks good
    and means nothing.
    """


class NetworkForbiddenError(ProviderError):
    """A fetch was attempted in a mode where the cache is the only source."""


@dataclass(frozen=True)
class ProviderSpec:
    """What a provider is and what it is allowed to do."""

    name: str

    #: Can it answer "what was true on date X" rather than "what is true now"?
    #: EDGAR can (every fact carries its `filed` date). An RSS feed cannot.
    supports_point_in_time: bool

    #: Minimum seconds between calls. 0 means unlimited.
    min_interval_s: float = 0.0

    #: Does serving a request involve an outbound HTTP call? False for a
    #: cache-only reader such as the IB bars bridge -- which is why that one is
    #: usable from `decide` at all.
    reads_network: bool = True

    #: Free-text, surfaced in `desk providers`. Keeps the §7.3 reasoning next
    #: to the code rather than only in the design doc.
    role: str = ""
