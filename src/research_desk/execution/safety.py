"""Runtime safety assertions.

Carried **near-verbatim** from the ORB+GEX engine. ``assert_paper_account`` is
kept byte-identical on purpose: it is the single most safety-critical function in
either codebase, and divergence between the two copies is the actual risk
(migration plan §1, architecture §15.11). A security-relevant fix here must be
applied in the ORB repo too, and **nothing will remind you**.

Only ``assert_can_trade`` changed on the way across -- it now reads the new
``RunMode`` rather than the options engine's ``DataMode``. Both replay modes
return ``can_trade = False``.

The port number is not a safety guarantee. A socat mapping, an env var, or a
reconnect landing on a different Gateway can all point a nominally-paper setup at
a live account. ``ib.managedAccounts()`` is the only thing that actually knows,
so that is what we gate on.
"""

from __future__ import annotations

import logging

from ib_async import IB

from ..models.modes import RunMode

logger = logging.getLogger(__name__)

# IB paper accounts: DU = paper individual, DF = paper financial advisor.
PAPER_ACCOUNT_PREFIXES = ("DU", "DF")


class LiveAccountError(RuntimeError):
    """Raised when the connected account is not a paper account."""


class ReplayTradeError(RuntimeError):
    """Raised when something tries to trade on replayed historical prices."""


def assert_paper_account(ib: IB, *, config_is_paper: bool = True) -> list[str]:
    """Verify every managed account is a paper account.

    Call this immediately after connecting *and* immediately before every
    ``placeOrder``. It costs microseconds, and the second call is the one that
    catches a reconnect that landed somewhere unexpected.
    """
    accounts = ib.managedAccounts()

    if not accounts:
        raise LiveAccountError(
            "No managed accounts reported by IB. Refusing to trade: cannot prove "
            "this is a paper account."
        )

    live = [a for a in accounts if not a.startswith(PAPER_ACCOUNT_PREFIXES)]
    if live:
        raise LiveAccountError(
            f"Non-paper account(s) present: {live}. Paper accounts start with "
            f"{' or '.join(PAPER_ACCOUNT_PREFIXES)}. Refusing to trade."
        )

    if not config_is_paper:
        raise LiveAccountError(
            "Config declares account.type != paper. This engine is paper-only."
        )

    return accounts


def assert_can_trade(mode: RunMode) -> None:
    """Reject order placement when the decision came from replayed prices.

    Enforced here rather than only in the caller so a future caller cannot route
    around it by constructing a broker client directly.
    """
    if not mode.can_trade:
        raise ReplayTradeError(
            f"RUN_MODE={mode.value} cannot place orders: the prices driving "
            "this decision are historical, so the order would be meaningless."
        )
