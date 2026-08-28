"""Portfolio intent, sizing, and the Python veto. **Stage 6.**

``PortfolioIntent`` is consumed through **four distinct channels, deliberately
not one** (architecture §8):

1. **Deterministic pre-filter**, no LLM -- universe minus exclusions minus
   earnings blackout minus at-max-positions, computed before any model runs.
2. **Drift table**, no LLM -- ``engine.compute_gaps()`` hands the Trader a table
   of current vs target weight per symbol and theme. Its job is *choosing which
   gap to close*, not inventing allocations. This is the biggest reliability win
   in the design: it converts an open-ended "how much should we buy" into a
   bounded "close this gap or don't", which small models handle far better.
3. **Prompt context**, LLM -- objective, theme theses and free-text constraints,
   into the **Trader and Fund Manager system prompts only**.
4. **Hard veto**, no LLM -- everything under ``risk:`` is enforced by
   ``compliance.py`` *after* sizing. Any violation blocks the order regardless of
   what the fund manager decided.

> **Analysts are intentionally intent-blind, and this is not an oversight.** Do
> not tell the fundamentals analyst "we are bullish on AI infrastructure" before
> it reads the 10-K. The entire value of a bear researcher evaporates if every
> upstream report was primed with your thesis. Intent enters at the Trader node
> and no earlier. A future refactor will want to "helpfully" fix this; there is a
> test asserting intent text is absent from analyst prompts.

Two-key system: the LLM proposes, Python vetoes. That is where trust comes from.
"""
