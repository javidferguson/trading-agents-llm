"""``NodeContext`` -- everything a node needs that is not graph state.

The split matters. ``DecisionState`` is the *run*: it is serialised to JSONL,
replayed, and diffed at Stage 8. ``NodeContext`` is the *machinery*: clients,
config, budget. Putting an HTTP client in the state would make the state
unserialisable; putting the symbol in the context would make replay a lie.

Nodes therefore keep the signature architecture §1 makes mandatory::

    async def node(state: DecisionState, ctx: NodeContext) -> NodePatch

``graph/build.py`` binds ``ctx`` at assembly time, so LangGraph only ever sees a
one-argument callable and the node body never touches a framework type.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .config import Settings
from .models.modes import RunMode


def new_run_id() -> str:
    """A run id that sorts chronologically, which is what you want in a log dir."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class NodeContext:
    """Immutable per-run machinery handed to every node."""

    settings: Settings
    run_id: str = field(default_factory=new_run_id)
    mode: RunMode = RunMode.LIVE
    #: The point-in-time anchor. Every provider call in this run is pinned to it,
    #: which is what keeps look-ahead bias out of replay (architecture §7.4).
    as_of: date = field(default_factory=date.today)
    prompt_pack_version: str = "0"
    config_hash: str = ""
    started_at: datetime = field(default_factory=datetime.now)

    #: Escape hatch for things that are not yet their own field. Prefer adding a
    #: typed field when a stage lands rather than letting this become a bag.
    extras: dict[str, Any] = field(default_factory=dict)

    # Arriving with later stages, each as a typed field on this dataclass:
    #   Stage 1: llm       -- LLMRouter, profiles -> provider clients
    #   Stage 2: providers -- ProviderRegistry, the single fetch entry point
    #   Stage 5: budget    -- RunBudget(max_llm_calls, max_wall_s, max_usd)
    #   Stage 9: memory    -- MemoryStore

    @property
    def elapsed_s(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()
