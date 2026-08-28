"""Run mode, with the safety behaviour attached to the type.

Carried as a *pattern* from the ORB+GEX engine's ``DataMode`` (migration plan
§1, "carried as pattern, not as code"). The point of putting ``can_trade`` on
the enum rather than comparing strings at call sites is that a new call site
cannot forget the rule -- it has to ask the type.

``str`` base so that ``mode`` serialises to ``"live"`` / ``"replay"`` /
``"replay_llm"`` in ``DecisionState`` JSONL without a custom encoder.
"""

from __future__ import annotations

from enum import Enum


class RunMode(str, Enum):
    """How this run gets its data and its model responses."""

    #: Fetch live, call models for real, and (in ``execute``) place orders.
    LIVE = "live"

    #: Re-run a past date against frozen cached data with a different model.
    REPLAY = "replay"

    #: Re-run against *recorded LLM responses* to test downstream changes at
    #: zero cost and zero latency. Use this constantly during development.
    REPLAY_LLM = "replay_llm"

    @property
    def can_trade(self) -> bool:
        """Only a live run may place an order.

        Both replay modes are driven by historical prices, so an order would be
        meaningless. ``execution.safety.assert_can_trade`` enforces this inside
        the order path, not merely at the call site.
        """
        return self is RunMode.LIVE

    @property
    def allows_network(self) -> bool:
        """In replay the cache is the *only* permitted source.

        Architecture §7.4. Look-ahead bias is the number one way agentic
        backtests produce fictional Sharpe ratios, and the cheapest guard is to
        make an HTTP call in replay impossible rather than merely discouraged.
        """
        return self is RunMode.LIVE

    @property
    def uses_recorded_llm(self) -> bool:
        """Whether the router should replay ``LLMCallRecord.raw_response``."""
        return self is RunMode.REPLAY_LLM
