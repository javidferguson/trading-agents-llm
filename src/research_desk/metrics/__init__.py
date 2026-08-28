"""All arithmetic. **Stage 2.**

> **Never let an LLM do arithmetic** (architecture §2, §15.3).

RSI, ATR, drawdown, YoY growth, percentiles, position sizing -- every number is
computed here and handed to the model as a fact. An 8B model will confidently
produce a wrong RSI. It is genuinely good at "RSI is 71 and price is at the upper
Bollinger band -- that's stretched." This single rule is the difference between
local models being usable and useless here.

The rule extends to *prompts*, not just code. Sentiment goes in as a percentile
and a direction of change, never as an adjective: tell an 8B model "sentiment is
very positive" and it will say BUY, confidently, every time -- while the
better-documented short-horizon reading of extreme bullish sentiment is
*contrarian*. Give it the number; let the bull and bear researchers argue
(architecture §7.1).

Planned: ``indicators.py`` (the full §7.1 metric set), ``regime.py`` (the twelve
buckets that memory retrieval keys on), ``snapshot.py``
(``build_market_snapshot()``).
"""
