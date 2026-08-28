"""Carried with ``execution/safety.py``.

Migration plan §1: *"A carried safety function with no carried test is worse than
not carrying it."*

``assert_paper_account`` is the single most safety-critical function in either
codebase. It is kept byte-identical to the ORB+GEX copy on purpose, and the real
risk is the two copies drifting -- a security-relevant fix here must be applied
there too, and **nothing will remind you** (architecture §15.11).
"""

from __future__ import annotations

import pytest

from research_desk.execution.safety import (
    LiveAccountError,
    ReplayTradeError,
    assert_can_trade,
    assert_paper_account,
)
from research_desk.models.modes import RunMode


class FakeIB:
    """Just enough IB to exercise the gate. The real one is not needed."""

    def __init__(self, accounts: list[str]):
        self._accounts = accounts

    def managedAccounts(self) -> list[str]:  # noqa: N802 - mirrors ib_async
        return self._accounts


def test_paper_accounts_pass() -> None:
    assert assert_paper_account(FakeIB(["DU1234567"])) == ["DU1234567"]
    assert assert_paper_account(FakeIB(["DF7654321"])) == ["DF7654321"]


def test_live_account_is_refused() -> None:
    with pytest.raises(LiveAccountError, match="Non-paper account"):
        assert_paper_account(FakeIB(["U1234567"]))


def test_one_live_account_among_paper_ones_is_refused() -> None:
    """The port is not a safety guarantee; a mixed set is the reconnect case."""
    with pytest.raises(LiveAccountError, match="Non-paper account"):
        assert_paper_account(FakeIB(["DU1234567", "U7654321"]))


def test_no_accounts_is_refused_rather_than_allowed() -> None:
    """Absence of evidence is not evidence of a paper account."""
    with pytest.raises(LiveAccountError, match="cannot prove"):
        assert_paper_account(FakeIB([]))


def test_config_declaring_non_paper_is_refused_even_with_paper_accounts() -> None:
    with pytest.raises(LiveAccountError, match="paper-only"):
        assert_paper_account(FakeIB(["DU1234567"]), config_is_paper=False)


# --------------------------------------------------------------------------- #
# assert_can_trade -- the one thing that changed on the way across
# --------------------------------------------------------------------------- #


def test_live_mode_may_trade() -> None:
    assert_can_trade(RunMode.LIVE)


@pytest.mark.parametrize("mode", [RunMode.REPLAY, RunMode.REPLAY_LLM])
def test_replay_modes_may_not_trade(mode: RunMode) -> None:
    """Both replay modes are driven by historical prices, so an order is meaningless."""
    with pytest.raises(ReplayTradeError, match=mode.value):
        assert_can_trade(mode)


def test_every_mode_is_covered() -> None:
    """A new RunMode must make a deliberate decision about trading.

    Without this, adding a mode later silently defaults it to whatever
    `can_trade` happens to return, which is exactly the kind of quiet default
    that safety code should not have.
    """
    decided = {RunMode.LIVE, RunMode.REPLAY, RunMode.REPLAY_LLM}
    assert set(RunMode) == decided, (
        "A RunMode was added without deciding whether it can trade. "
        "Update safety.assert_can_trade and this test together."
    )
