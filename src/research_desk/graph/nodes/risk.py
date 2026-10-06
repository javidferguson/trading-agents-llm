"""The risk committee (§3 nodes 10-12) and the fund manager (node 13).

Same four termination conditions as the research debate, reusing the same
helpers -- `_score_novelty` and `_stop_reason` are parameterised by which
transcript and which config key, so there is one implementation of §5's rules
rather than two that drift apart.

The trio run **sequentially**, not as a fan-out: risky, then neutral, then
safe, each seeing what came before. That is what makes it a debate rather than
three parallel opinions, and the neutral member's whole job is arbitrating
between the other two.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ...context import NodeContext
from ...llm.structured import structured
from ...models.state import (
    DebateTranscript,
    DebateTurn,
    DecisionState,
    NodeError,
    NodePatch,
    RiskVerdict,
)
from ...prompts import load_prompt
from .researchers import _score_novelty, _stop_reason, render_reports

logger = logging.getLogger(__name__)

#: speaker -> node name in models.yaml, in the order they argue.
RISK_TRIO = (("risky", "risky_debater"),
             ("neutral", "neutral_debater"),
             ("safe", "safe_debater"))


def render_proposal(state: DecisionState) -> str:
    """The trader's proposal, for the risk committee to argue about."""
    proposal = state.trader_proposal
    if proposal is None:
        return "TRADER PROPOSAL: none was produced.\n"
    lines = [
        "TRADER PROPOSAL",
        f"  action           {proposal.action}",
        f"  target weight    {proposal.target_weight_pct}% of equity",
        f"  conviction       {proposal.conviction}",
        f"  horizon          {proposal.horizon_days} days",
        f"  rationale        {' '.join(proposal.rationale.split())}",
        f"  invalidation     {' '.join(proposal.invalidation.split())}",
        f"  trader's own counterargument",
        f"                   {' '.join(proposal.strongest_counterargument.split())}",
    ]
    if proposal.parse_failed:
        lines.append(
            "  NOTE: this proposal is DEGRADED -- the trader produced no usable "
            "output. There is nothing here to approve."
        )
    return "\n".join(lines) + "\n"


def render_research(state: DecisionState) -> str:
    """The research verdict, condensed for the risk committee."""
    verdict = state.research_verdict
    if verdict is None:
        return "RESEARCH VERDICT: none was reached.\n"
    lines = [
        "RESEARCH VERDICT",
        f"  winner     {verdict.winner} @ {verdict.confidence}",
        f"  thesis     {' '.join(verdict.thesis.split())}",
        f"  strongest counterargument",
        f"             {' '.join(verdict.strongest_counterargument.split())}",
        f"  debate     {state.research_debate.rounds_completed} round(s), "
        f"stopped: {state.research_debate.stop_reason}",
    ]
    return "\n".join(lines) + "\n"


def render_risk_debate(transcript: DebateTranscript) -> str:
    if not transcript.turns:
        return "RISK COMMITTEE SO FAR: nothing yet; you are opening.\n"
    lines = ["RISK COMMITTEE SO FAR", ""]
    for turn in transcript.turns:
        lines.append(f"  round {turn.round} -- {turn.speaker}")
        lines.append(f"    {' '.join(turn.claim.split())}")
        if turn.supporting_report_kinds:
            lines.append(f"    resting on: {', '.join(turn.supporting_report_kinds)}")
        lines.append("")
    return "\n".join(lines)


def make_risk_debater(speaker: str, node: str):
    """Build one member of the risk committee."""

    async def debater(state: DecisionState, ctx: NodeContext) -> NodePatch:
        router = ctx.extras.get("router")
        if router is None:
            return {"errors": [NodeError(node=node, kind="misconfigured",
                                         message="no LLM router on the context")]}

        budget = ctx.extras.get("budget")
        if budget is not None and (spent := budget.exhausted()):
            return {"notes": [f"{node}: skipped, budget exhausted ({spent})"]}

        # Nothing to argue about. A committee debating a degraded proposal is
        # three model calls spent deciding how to size a trade that does not
        # exist.
        if state.trader_proposal is None or state.trader_proposal.parse_failed:
            return {"notes": [f"{node}: skipped, no usable proposal to assess"]}

        round_number = state.risk_debate.rounds_completed + 1
        result = await structured(
            router, node, DebateTurn,
            [
                {"role": "system", "content": load_prompt(node)},
                {"role": "user", "content": (
                    f"{render_proposal(state)}\n"
                    f"{render_research(state)}\n"
                    f"{render_reports(state)}\n"
                    f"{render_risk_debate(state.risk_debate)}\n"
                    f"This is round {round_number}. Make your argument about the "
                    f"risk of this {state.symbol} proposal."
                )},
            ],
            degraded_fields={"round": round_number, "speaker": speaker},
        )
        if budget is not None:
            budget.record(calls=len(result.records), usd=result.usd)

        turn = result.value.model_copy(
            update={"round": round_number, "speaker": speaker}
        )
        return {
            "risk_turns": [turn],
            "llm_calls": result.records,
            "notes": [f"{node} r{round_number}: {turn.claim[:70]}"],
        }

    debater.__name__ = node
    return debater


async def fund_manager(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Approve, adjust or veto (§3 node 13). The last judgement before a human."""
    node = "fund_manager"
    router = ctx.extras.get("router")
    if router is None:
        return {"errors": [NodeError(node=node, kind="misconfigured",
                                     message="no LLM router on the context")]}

    round_number = state.risk_debate.rounds_completed + 1
    this_round = [t for t in state.risk_turns if t.round == round_number]
    scored = _score_novelty(state, this_round, transcript=state.risk_debate)

    budget = ctx.extras.get("budget")
    if budget is not None and (spent := budget.exhausted()):
        # §5: budget always wins, and a run that cannot afford a decision
        # degrades to HOLD rather than crashing.
        verdict = RiskVerdict.degraded(f"budget exhausted before the decision: {spent}")
        return _close(state, scored, round_number, "budget", verdict,
                      f"{node}: skipped, budget exhausted ({spent})")

    if state.trader_proposal is None or state.trader_proposal.parse_failed:
        verdict = RiskVerdict.degraded("the trader produced no usable proposal")
        return _close(state, scored, round_number, "error", verdict,
                      f"{node}: vetoed, no usable proposal")

    result = await structured(
        router, node, RiskVerdict,
        [
            {"role": "system", "content": load_prompt(node)},
            {"role": "user", "content": (
                f"{render_proposal(state)}\n"
                f"{render_research(state)}\n"
                f"{render_risk_debate(_with(state, scored))}\n"
                f"{render_reports(state)}\n"
                f"Decide on the {state.symbol} proposal."
            )},
        ],
    )
    if budget is not None:
        budget.record(calls=len(result.records), usd=result.usd)

    verdict = result.value
    stop = _stop_reason(state, ctx, scored, None, round_number,
                        transcript=state.risk_debate,
                        rounds_key="max_risk_rounds")

    patch = _close(
        state, scored, round_number, stop or "max_rounds", verdict,
        f"{node} r{round_number}: {verdict.decision} -> {verdict.action} "
        f"{verdict.target_weight_pct:.1f}% @ {verdict.conviction:.2f}"
        + (" (degraded)" if result.degraded else ""),
    )
    patch["llm_calls"] = result.records
    if result.degraded:
        patch["errors"] = [NodeError(node=node, kind="degraded",
                                     message=verdict.degraded_reason or "unknown")]
    return patch


def _with(state: DecisionState, turns: list[DebateTurn]) -> DebateTranscript:
    return DebateTranscript(
        turns=[*state.risk_debate.turns, *turns],
        rounds_completed=state.risk_debate.rounds_completed,
    )


def _close(
    state: DecisionState,
    turns: list[DebateTurn],
    round_number: int,
    stop: str,
    verdict: RiskVerdict,
    note: str,
) -> NodePatch:
    return {
        "risk_debate": DebateTranscript(
            turns=[*state.risk_debate.turns, *turns],
            rounds_completed=round_number,
            stop_reason=stop,
        ),
        "risk_verdict": verdict,
        "notes": [note],
    }
