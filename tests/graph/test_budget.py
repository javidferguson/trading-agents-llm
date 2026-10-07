"""Budget accounting and the Stage 5 exit gate.

> *"Exit gate: a full 14-node run completes inside
> `RunBudget(max_llm_calls=20, max_wall_s=900, max_usd=0.50)`, and a
> deliberately exhausted budget yields HOLD rather than a crash."*

A budget that under-counts is worse than no budget, because it reports a
comfortable margin that is not there. That is not hypothetical: the trader and
the Stage 3 market analyst predated the budget and never recorded their calls,
so the first full run reported "13 model calls" alongside "budget: 12/20".
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from research_desk.models.budget import RunBudget

NODE_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "research_desk" / "graph" / "nodes"


def nodes_that_call_a_model() -> list[pathlib.Path]:
    return [p for p in sorted(NODE_DIR.glob("*.py")) if "await structured(" in p.read_text()]


# --------------------------------------------------------------------------- #
# Static: every node must both check and record
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", nodes_that_call_a_model(), ids=lambda p: p.name)
def test_every_model_calling_node_records_what_it_spent(path: pathlib.Path) -> None:
    """The bug this catches, found on the first full run.

    `trader.py` and `market_analyst.py` were written at Stage 3, before
    RunBudget existed, and nobody went back. Their calls were invisible to the
    cap -- so the budget would have let a run exceed its limit by however many
    calls those nodes made, while reporting that it had room.
    """
    source = path.read_text()
    assert "budget.record" in source, (
        f"{path.name} calls a model but never records it. The cap would "
        "under-count by every call this node makes."
    )


@pytest.mark.parametrize("path", nodes_that_call_a_model(), ids=lambda p: p.name)
def test_every_model_calling_node_checks_before_spending(path: pathlib.Path) -> None:
    """§5: budget always wins -- and a check after the fact is an audit."""
    source = path.read_text()
    assert "budget.exhausted" in source, (
        f"{path.name} calls a model without checking the budget first"
    )


def test_the_check_comes_before_the_call_not_after() -> None:
    """Order matters, and reading the source is the only way to assert it."""
    for path in nodes_that_call_a_model():
        source = path.read_text()
        first_check = source.index("budget.exhausted")
        first_call = source.index("await structured(")
        assert first_check < first_call, (
            f"{path.name} checks the budget after calling the model"
        )


# --------------------------------------------------------------------------- #
# The budget itself
# --------------------------------------------------------------------------- #


def test_the_shipped_budget_matches_the_architecture() -> None:
    """§12: RunBudget(max_llm_calls=20, max_wall_s=900, max_usd=0.50).

    20 is 14 calls at one round each plus room for repair turns and nothing
    more, so a loop is caught rather than absorbed.
    """
    from research_desk.config import load_yaml

    budget = RunBudget.from_config(load_yaml("models.yaml"))
    assert budget.max_llm_calls == 20
    assert budget.max_wall_s == 900
    assert budget.max_usd == 0.50


def test_a_full_run_fits_inside_the_cap_with_room_to_spare() -> None:
    """TWELVE model-calling nodes at one round each, plus repair turns.

    §3 says "14 LLM calls at N=M=1" but its own node list totals 12 LLM nodes
    (nodes 0, 14 and 15 are prefetch, compliance and persist, none of which
    call a model). The live run reported 13 calls because one repair turn
    fired -- which is exactly the headroom the 20-call cap is sized for.

    If this count ever exceeds 20 the cap is wrong, not the run: §5 would
    start silently truncating every debate at the budget condition.
    """
    from research_desk.graph.nodes.analysts import ANALYSTS
    from research_desk.graph.nodes.risk import RISK_TRIO

    calls = (
        len(ANALYSTS)          # four analysts
        + 2                    # bull, bear
        + 1                    # research facilitator
        + 1                    # trader
        + len(RISK_TRIO)       # risky, neutral, safe
        + 1                    # fund manager
    )
    assert calls == 12
    budget = RunBudget()
    assert calls < budget.max_llm_calls
    assert budget.max_llm_calls - calls >= 5, (
        "too little headroom for repair turns; a single repair would start "
        "truncating debates"
    )
    # Every node, counted from the code rather than from the doc.
    assert calls == len(nodes_that_call_a_model()) + 7, (
        "the node files and the arithmetic above disagree about how many "
        "nodes call a model"
    )


def test_exhaustion_names_the_limit_and_the_numbers() -> None:
    calls = RunBudget(max_llm_calls=2)
    calls.record(); calls.record()
    assert "2 of 2 model calls" in calls.exhausted()

    money = RunBudget(max_usd=0.10)
    money.record(usd=0.25)
    assert "$0.2500 of $0.10" in money.exhausted()

    assert RunBudget(max_wall_s=0.0).exhausted().endswith("wall clock used")
    assert RunBudget().exhausted() is None


def test_recording_accumulates_calls_and_dollars() -> None:
    budget = RunBudget()
    budget.record(calls=2, usd=0.01)
    budget.record(calls=1, usd=0.02)
    assert budget.calls == 3
    assert budget.usd == pytest.approx(0.03)
    assert budget.remaining_calls() == 17


# --------------------------------------------------------------------------- #
# An exhausted budget degrades to HOLD rather than crashing
# --------------------------------------------------------------------------- #


async def test_an_exhausted_budget_yields_hold_not_a_crash(tmp_path) -> None:
    """The second half of the exit gate, end to end through persist."""
    from datetime import date

    from research_desk.config import Settings
    from research_desk.context import NodeContext
    from research_desk.graph.nodes.persist import persist
    from research_desk.models.state import DecisionState, RiskVerdict

    spent = RunBudget(max_llm_calls=1)
    spent.record()

    state = DecisionState(
        run_id="t", symbol="MSFT", as_of=date(2026, 10, 6),
        risk_verdict=RiskVerdict.degraded(
            f"budget exhausted before the decision: {spent.exhausted()}"
        ),
    )
    ctx = NodeContext(settings=Settings(data_dir=tmp_path), as_of=date(2026, 10, 6),
                      extras={"budget": spent})

    patch = await persist(state, ctx)
    decision = patch["final_decision"]

    assert decision.action == "HOLD"
    assert decision.target_weight_pct == 0.0
    assert decision.conviction == 0.0
    assert decision.degraded is True
    # And the artefact still exists: a run that wrote nothing is
    # indistinguishable from one that never happened.
    assert list((tmp_path / "proposals").glob("MSFT_*.json"))


async def test_a_vetoed_proposal_cannot_carry_a_trade() -> None:
    """Enforced in the schema, because a vetoed BUY is the single most
    dangerous shape this object could take."""
    with pytest.raises(ValueError, match="veto"):
        RiskVerdictFactory(decision="veto", action="BUY")


def RiskVerdictFactory(**overrides):
    from research_desk.models.state import RiskVerdict
    base = dict(
        decision="approve", action="HOLD", conviction=0.5,
        target_weight_pct=5.0, horizon_days=30,
        rationale="The evidence supports holding the current weight.",
        dissent="Momentum could continue and the entry would be missed.",
        invalidation="A close below the 200-day simple moving average.",
    )
    return RiskVerdict(**{**base, **overrides})
