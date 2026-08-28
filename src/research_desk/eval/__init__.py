"""Replay, baselines and scoring. **Stage 8 -- the go/no-go.**

The purpose is *not* to prove alpha; on three months you cannot. It is to detect
disqualifying badness, check calibration, and answer one question: **does the
multi-agent structure beat a single LLM call?**

| | Baseline |
|---|---|
| B0 | Buy-and-hold SPY |
| B1 | Always HOLD |
| B2 | Random, turnover-matched, 200 seeds -- a null *distribution*, not a point |
| B3 | Deterministic rule (50/200 SMA + RSI) |
| **B4** | **Single LLM call, same snapshot, one prompt** |
| A | The full 14-node pipeline |

> **If A does not beat B4 by a margin outside B2's noise band, the multi-agent
> layer is not earning its 14x cost -- ship B4.** Build the harness so that
> switching A<->B4 is a one-line config change, because a comparison that takes
> effort to re-run is a comparison you will stop running. Most multi-agent
> systems fail this test, and the plan is built so that failing it still leaves
> something good.

Two reporting rules that decide whether the numbers mean anything:

* **Report residual return vs SPY**, not raw, or you will conclude the agents are
  geniuses in an up market.
* **Report the survivorship-bias caveat next to every number.** Every free source
  lists companies that exist today, so any universe assembled now silently
  excludes everything that went to zero. It is not a footnote; it is the reason a
  plausible-looking equity curve may be fiction (architecture §7.5, §15.6).

Shadow mode is the primary tool: run ``decide`` daily on live data with
``execute`` never called, and accumulate 60-90 sessions. Out-of-sample and
leakage-free -- worth more than any replay.
"""
