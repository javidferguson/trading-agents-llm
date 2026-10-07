"""``RunBudget`` -- the hard cap. **Budget always wins (§5).**

> *"Conditions 1 and 4 are hard caps enforced in Python. Never let the model
> alone decide when to stop."*

A debate that decides for itself when it is finished will, with a local model,
restate itself politely for as long as you let it. Three independent limits,
any one of which ends the run:

* **calls** -- 14 at N=M=1, so 20 leaves room for repair turns and nothing more.
* **wall clock** -- at ~30 s per Ollama call a runaway loop is minutes of
  silence before you notice.
* **dollars** -- only the fund manager is hosted, but §12's estimate is
  "~$0.03 per decision", so $0.50 is a bug detector rather than a limit.

Exhaustion is not an error: it ends the debate with ``stop_reason="budget"``
and the run degrades to HOLD (§5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunBudget:
    """Mutable spend tracker for one decision run."""

    max_llm_calls: int = 20
    max_wall_s: float = 900.0
    max_usd: float = 0.50

    calls: int = 0
    usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_config(cls, config: dict | None) -> "RunBudget":
        body = (config or {}).get("budget") or {}
        return cls(
            max_llm_calls=int(body.get("max_llm_calls", 20)),
            max_wall_s=float(body.get("max_wall_s", 900.0)),
            max_usd=float(body.get("max_usd", 0.50)),
        )

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def record(self, calls: int = 1, usd: float = 0.0) -> None:
        self.calls += calls
        self.usd += usd

    def exhausted(self) -> str | None:
        """Why the budget is spent, or ``None``. A reason, not a boolean.

        "budget" alone in a stop_reason tells you nothing at Stage 8; "20 of 20
        calls" tells you whether to raise the cap or fix a loop.
        """
        if self.calls >= self.max_llm_calls:
            return f"{self.calls} of {self.max_llm_calls} model calls used"
        if self.elapsed_s >= self.max_wall_s:
            return f"{self.elapsed_s:.0f}s of {self.max_wall_s:.0f}s wall clock used"
        if self.usd >= self.max_usd:
            return f"${self.usd:.4f} of ${self.max_usd:.2f} spent"
        return None

    def remaining_calls(self) -> int:
        return max(self.max_llm_calls - self.calls, 0)
