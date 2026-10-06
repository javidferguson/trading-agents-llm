"""THE STAGE 4 EXIT GATE.

> *"Exit gate: the debate stops for the right reason every time, and the
> `stop_reason` is correct. Force each of the four conditions in a test."*

The four, from §5:

1. ``rounds_completed >= cfg.max_rounds``   -- a hard cap in Python
2. facilitator returns ``converged=True``   -- the only one the model influences
3. the novelty guard                        -- nobody said anything new
4. ``budget.exhausted()``                   -- **budget always wins**

Priority matters as much as the conditions. §5: *"Conditions 1 and 4 are hard
caps enforced in Python. Never let the model alone decide when to stop."* So a
facilitator demanding another round it cannot afford does not get one, and
that ordering is tested explicitly rather than assumed.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.graph.nodes.researchers import (
    _score_novelty,
    _stop_reason,
    research_facilitator,
)
from research_desk.llm.router import LLMResponse, LLMRouter, ModelProfile
from research_desk.models.budget import RunBudget
from research_desk.models.state import (
    AnalystReport,
    DebateTranscript,
    DebateTurn,
    DecisionState,
    ResearchVerdict,
)

AS_OF = date(2026, 10, 6)


def verdict_json(*, converged: bool, new_info: bool = True, reason: str = "") -> str:
    return json.dumps({
        "winner": "tie",
        "thesis": "The evidence is genuinely balanced between the two cases.",
        "strongest_counterargument":
            "The bull case rests on one report that may be stale.",
        "confidence": 0.5,
        "converged": converged,
        "converged_reason": reason,
    })


class Scripted:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def complete(self, profile, messages, schema=None):
        self.calls += 1
        return LLMResponse(text=self.reply, model="m", profile=profile.name,
                           provider=profile.provider)

    async def aclose(self):
        return None


def router_for(reply: str) -> tuple[LLMRouter, Scripted]:
    client = Scripted(reply)
    return (
        LLMRouter(
            profiles={"deep": ModelProfile(name="deep", provider="s", model="m")},
            node_routes={"research_facilitator": "deep"},
            clients={"s": client},
        ),
        client,
    )


def ctx_with(*, budget: RunBudget | None = None, max_rounds: int = 1,
             router: LLMRouter | None = None) -> NodeContext:
    extras: dict = {"max_research_rounds": max_rounds}
    if budget is not None:
        extras["budget"] = budget
    if router is not None:
        extras["router"] = router
    return NodeContext(settings=Settings(), as_of=AS_OF, extras=extras)


def state_with(turns: list[DebateTurn], *, completed: int = 0,
               previous: list[DebateTurn] | None = None) -> DecisionState:
    return DecisionState(
        run_id="t", symbol="MSFT", as_of=AS_OF,
        analyst_reports={"market": AnalystReport(
            kind="market", stance="neutral", confidence=0.5,
            summary="Balanced.", key_points=["a", "b", "c"],
        )},
        debate_turns=turns,
        research_debate=DebateTranscript(
            turns=previous or [], rounds_completed=completed
        ),
    )


def turn(speaker: str, claim: str, *, round: int = 1, new: bool = True) -> DebateTurn:
    return DebateTurn(round=round, speaker=speaker, claim=claim,
                      supporting_report_kinds=["market"], new_information=new)


# --------------------------------------------------------------------------- #
# Condition 1 -- max_rounds. A hard cap.
# --------------------------------------------------------------------------- #


async def test_condition_1_max_rounds() -> None:
    """Default is 1, not 2 -- §5 says start cheap."""
    router, _ = router_for(verdict_json(converged=False))
    turns = [turn("bull", "The trend is intact above the 200-day."),
             turn("bear", "Twelve-month momentum is negative.")]

    patch = await research_facilitator(
        state_with(turns), ctx_with(max_rounds=1, router=router))

    assert patch["research_debate"].stop_reason == "max_rounds"
    assert patch["research_debate"].rounds_completed == 1


async def test_max_rounds_allows_a_second_round_when_configured() -> None:
    """Guard against the cap being the only thing that ever stops it."""
    router, _ = router_for(verdict_json(converged=False))
    turns = [turn("bull", "Gross profitability is 0.28 and accruals are negative."),
             turn("bear", "The 52-week percentile is 0.94 and RSI is 68.")]

    patch = await research_facilitator(
        state_with(turns), ctx_with(max_rounds=2, router=router))
    assert patch["research_debate"].stop_reason is None, "should continue"


# --------------------------------------------------------------------------- #
# Condition 2 -- the facilitator converged. Advisory, and checked LAST.
# --------------------------------------------------------------------------- #


async def test_condition_2_converged() -> None:
    router, _ = router_for(
        verdict_json(converged=True, reason="positions are clear and opposed")
    )
    turns = [turn("bull", "Buybacks shrank the share count by 0.09%."),
             turn("bear", "Relative strength against the sector is -39% over a year.")]

    patch = await research_facilitator(
        state_with(turns), ctx_with(max_rounds=3, router=router))

    assert patch["research_debate"].stop_reason == "converged"
    assert patch["research_verdict"].converged_reason


async def test_a_converged_facilitator_cannot_override_the_budget() -> None:
    """§5: budget always wins. Condition 4 is checked before condition 2."""
    spent = RunBudget(max_llm_calls=1)
    spent.record()
    router, _ = router_for(verdict_json(converged=True, reason="done"))

    patch = await research_facilitator(
        state_with([turn("bull", "x y z w")]),
        ctx_with(budget=spent, max_rounds=3, router=router),
    )
    assert patch["research_debate"].stop_reason == "budget"


# --------------------------------------------------------------------------- #
# Condition 3 -- the novelty guard. Not optional (§5).
# --------------------------------------------------------------------------- #


async def test_condition_3_no_novelty_when_nobody_adds_anything() -> None:
    router, _ = router_for(verdict_json(converged=False))
    previous = [turn("bull", "Old point.", round=1),
                turn("bear", "Other old point.", round=1)]
    turns = [turn("bull", "Old point.", round=2, new=False),
             turn("bear", "Other old point.", round=2, new=False)]

    patch = await research_facilitator(
        state_with(turns, completed=1, previous=previous),
        ctx_with(max_rounds=5, router=router),
    )
    assert patch["research_debate"].stop_reason == "no_novelty"


def test_jaccard_overrides_a_too_agreeable_facilitator() -> None:
    """Two signals, because each catches what the other misses.

    A facilitator that calls a reworded repeat "new" is overruled by the
    overlap score, which cannot be talked out of its answer.
    """
    claim = "RSI is 68.8 and price is 22.6% above the 200-day SMA"
    reworded = "Price is 22.6% above the 200-day SMA and RSI is 68.8"

    previous = [turn("bull", claim, round=1)]
    state = state_with([turn("bull", reworded, round=2, new=True)],
                       completed=1, previous=previous)

    scored = _score_novelty(state, state.debate_turns)
    assert scored[0].new_information is False, (
        "the facilitator said this was new; the overlap score says otherwise"
    )


def test_a_genuinely_new_point_survives_both_checks() -> None:
    previous = [turn("bull", "RSI is 68.8 and price is above the 200-day SMA.", round=1)]
    state = state_with(
        [turn("bull", "Accruals are negative at -0.054, so earnings are cash-backed.",
              round=2, new=True)],
        completed=1, previous=previous,
    )
    assert _score_novelty(state, state.debate_turns)[0].new_information is True


def test_an_opening_round_is_always_novel() -> None:
    """Observed live: told to be strict, the facilitator marked BOTH round-1
    turns `new_information=False`. Nothing can restate nothing, and at
    max_rounds=2 that would have fired no_novelty before either side had
    spoken twice -- ending a debate that had not started.
    """
    state = state_with([turn("bull", "Opening argument.", round=1, new=False)])
    assert _score_novelty(state, state.debate_turns)[0].new_information is True


# --------------------------------------------------------------------------- #
# Condition 4 -- budget. Always wins.
# --------------------------------------------------------------------------- #


async def test_condition_4_budget_calls() -> None:
    spent = RunBudget(max_llm_calls=5)
    for _ in range(5):
        spent.record()

    router, client = router_for(verdict_json(converged=False))
    patch = await research_facilitator(
        state_with([turn("bull", "a b c d")]),
        ctx_with(budget=spent, max_rounds=9, router=router),
    )

    assert patch["research_debate"].stop_reason == "budget"
    assert client.calls == 0, "an exhausted budget must stop the call, not audit it"


def test_budget_exhaustion_says_which_limit_and_by_how_much() -> None:
    """"budget" alone tells you nothing at Stage 8; "20 of 20 calls" tells you
    whether to raise the cap or fix a loop."""
    calls = RunBudget(max_llm_calls=2)
    calls.record(); calls.record()
    assert "2 of 2 model calls" in calls.exhausted()

    money = RunBudget(max_usd=0.10)
    money.record(usd=0.25)
    assert "$0.2500 of $0.10" in money.exhausted()

    clock = RunBudget(max_wall_s=0.0)
    assert "wall clock" in clock.exhausted()

    assert RunBudget().exhausted() is None


# --------------------------------------------------------------------------- #
# Priority, which is as load-bearing as the conditions
# --------------------------------------------------------------------------- #


def test_the_four_conditions_resolve_in_the_documented_order() -> None:
    """Budget, then max_rounds, then novelty, then the model's opinion."""
    stale = [turn("bull", "Same.", round=2, new=False)]
    converged = ResearchVerdict(
        winner="tie", thesis="Balanced evidence on both sides.",
        strongest_counterargument="One report may be stale and unrepresentative.",
        confidence=0.5, converged=True, converged_reason="clear",
    )
    state = state_with(stale, completed=1,
                       previous=[turn("bull", "Same.", round=1)])

    spent = RunBudget(max_llm_calls=1)
    spent.record()
    # Everything fires at once; budget must win.
    assert _stop_reason(state, ctx_with(budget=spent, max_rounds=1),
                        stale, converged, 2) == "budget"
    # Without the budget, the hard round cap beats both model-influenced ones.
    assert _stop_reason(state, ctx_with(max_rounds=2), stale, converged, 2) == "max_rounds"
    # Without the cap, novelty beats convergence.
    assert _stop_reason(state, ctx_with(max_rounds=9), stale, converged, 2) == "no_novelty"
    # And only then does the facilitator's view decide.
    fresh = [turn("bull", "A genuinely different point about accruals.", round=2)]
    assert _stop_reason(state, ctx_with(max_rounds=9), fresh, converged, 2) == "converged"


def test_a_debate_that_should_continue_returns_none() -> None:
    """The negative case: without it, "always stop" would pass every test."""
    fresh = [turn("bull", "Accruals are negative at -0.054.", round=1)]
    open_verdict = ResearchVerdict(
        winner="bull", thesis="The cash-backed earnings case is stronger.",
        strongest_counterargument="Twelve-month relative strength is deeply negative.",
        confidence=0.6, converged=False,
    )
    assert _stop_reason(state_with(fresh), ctx_with(max_rounds=3),
                        fresh, open_verdict, 1) is None


def test_every_stop_reason_in_the_schema_is_reachable() -> None:
    """Guard against a condition being renamed and quietly never firing."""
    import typing
    allowed = set(typing.get_args(
        typing.get_args(DebateTranscript.model_fields["stop_reason"].annotation)[0]
    ))
    assert allowed == {"max_rounds", "converged", "no_novelty", "budget", "error"}
