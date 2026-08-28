"""Our LLM client. **Stage 1.**

LangGraph orchestrates; it does not make model calls. A node body calls *our*
router, never a ``langchain_core`` chat model -- that is where the abstraction
tax actually lives, and it makes Ollama's ``format=<json_schema>`` awkward for
no gain (architecture §1, constraint 1).

Arriving here:

* ``router.py``     -- profiles from ``config/models.yaml`` -> provider clients.
* ``structured.py`` -- schema-constrained decode, validate, **one repair turn**
  with the validation error appended, then ``schema.degraded()`` with
  ``parse_failed=True``. A degraded analyst report beats a crashed 8-minute run.

Two things not to defer, because both are worthless retroactively:

1. The repair turn. It is the part that gets skipped and then desperately needed
   at Stage 4.
2. Recording ``LLMCallRecord`` with the **raw** response from the very first
   call. ``mode="replay_llm"`` cannot be backfilled.
"""
