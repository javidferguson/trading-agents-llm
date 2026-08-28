"""``DecisionState`` and the records that accumulate inside it.

**This is the Stage 0 subset**, not the finished §4 schema. What is here is what
Stage 0 actually exercises, plus the two accumulating record types that must be
correct from the very first LLM call. Analyst reports, debate transcripts,
proposals and the final decision arrive in Stages 3–5.

Two things are load-bearing already and should not be treated as scaffolding:

* ``LLMCallRecord.raw_response`` -- ``mode="replay_llm"`` is worthless
  retroactively, so the raw text gets recorded from call number one.
* The **reducer annotations** on ``llm_calls`` and ``errors``. Nodes that run in
  a parallel fan-out append to both. Without an add-reducer LangGraph applies
  last-write-wins and the four analysts silently overwrite each other's records.
  This is the single nastiest failure mode in the design because nothing errors
  -- you just quietly lose three of four. ``tests/graph/test_reducers.py``
  asserts it.

> **Rule (architecture §1).** This JSONL is the source of truth for replay and
> audit. The LangGraph checkpointer exists *only* for crash-resume within a
> single live run. Two mechanisms, two jobs. Do not merge them, and do not let a
> future refactor "simplify" by deleting the JSONL.
"""

from __future__ import annotations

import operator
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

from .modes import RunMode


class LLMCallRecord(BaseModel):
    """One model call, recorded in full enough detail to replay it."""

    node: str
    profile: str
    provider: str
    model: str
    #: Provider-reported model digest. Two runs of "qwen3:8b" are not
    #: necessarily the same weights; the digest is what makes a cache key honest.
    model_digest: str | None = None
    prompt_hash: str
    #: The RAW response text, before parsing. This is what `replay_llm` feeds
    #: back in. Never replace it with the parsed object.
    raw_response: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    usd: float = 0.0
    #: 1 on the first try, 2 on the repair turn (architecture §6).
    attempt: int = 1
    parse_failed: bool = False
    at: datetime = Field(default_factory=datetime.now)


class NodeError(BaseModel):
    """A node that failed. Recorded rather than raised, so the run degrades.

    Failure direction is always HOLD (architecture §5) -- never fail toward a
    trade.
    """

    node: str
    kind: str
    message: str
    at: datetime = Field(default_factory=datetime.now)


class DecisionState(BaseModel):
    """The graph state. Used directly as the LangGraph state type.

    Fields that accumulate need reducer annotations; everything else is
    last-write-wins.
    """

    schema_version: int = 1

    run_id: str
    symbol: str
    #: The decision date, and therefore the replay anchor. Every provider call
    #: made during this run is pinned to it.
    as_of: date
    created_at: datetime = Field(default_factory=datetime.now)
    mode: RunMode = RunMode.LIVE

    #: Hash of the resolved config and the prompt pack. Two runs that disagree
    #: on either are not comparable, and at Stage 8 that matters.
    config_hash: str = ""
    prompt_pack_version: str = "0"

    # --- accumulating: reducers are mandatory, see the module docstring -------
    llm_calls: Annotated[list[LLMCallRecord], operator.add] = Field(default_factory=list)
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)

    #: Free-form breadcrumbs. Present from Stage 0 so the toy graph has
    #: something to write; harmless to keep.
    notes: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- arriving in later stages --------------------------------------------
    # Stage 2: intent, portfolio, snapshot, regime
    # Stage 3: analyst_reports, trader_proposal, final_decision
    # Stage 4: research_debate, research_verdict
    # Stage 5: risk_debate
    # Stage 6: violations
    # Stage 9: memory_hits

    def degraded_reason(self) -> str | None:
        """Why this run should fail toward HOLD, or ``None`` if it is healthy."""
        if not self.errors:
            return None
        return "; ".join(f"{e.node}: {e.kind}" for e in self.errors)


#: What a node returns: a partial-state patch, never a mutated state object.
#: Keeping node bodies to this signature is what makes them unit-testable,
#: replayable, and portable if LangGraph is ever un-adopted (architecture §1).
NodePatch = dict[str, Any]

__all__ = [
    "DecisionState",
    "LLMCallRecord",
    "NodeError",
    "NodePatch",
    "RunMode",
]
