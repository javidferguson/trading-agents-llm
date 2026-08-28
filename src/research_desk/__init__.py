"""A multi-agent LLM research desk for daily equities and ETFs.

Two processes, communicating through a file, never a shared event loop::

    [ decide ]  agents + LLM + free data  ->  proposal.json  ->  [ execute ]
     no IB connection, no ib_async import                  human gate + ib_async

See ``tradingagents-architecture.md`` for why, and
``tradingagents-migration-plan.md`` for in what order.
"""

from __future__ import annotations

__version__ = "0.1.0"
