# Known gaps and follow-ups

Deliberate omissions, with what unblocks each. Nothing here is a stub pretending
to work: every provider that exists is verified against live data, and every
metric that cannot be computed says why in the snapshot itself.

## Blocked on a free API key

| What | Needs | Consequence today |
|---|---|---|
| **Finnhub** — company news, earnings calendar, EPS surprises, recommendation trends | `FINNHUB_API_KEY` (free, 60/min) | No events block. The earnings-blackout compliance check (§9) has no input, so Stage 6 will need this |
| **FRED** — VIXCLS, DGS10, DGS2, T10Y2Y, DFF, CPIAUCSL, UNRATE, BAMLH0A0HYM2, NFCI | `FRED_API_KEY` (free) | No macro block |
| **`regime.py`** — §7.2's twelve buckets | FRED, above | **No regime tag.** Two of its three axes are VIX and the curve. Memory retrieval (§3 node 1) keys on `(symbol, regime)`, so Stage 9 is blocked on this |

These were left out rather than written unverified: the two bugs in
`scripts/fetch_bars.py` were both in code that had never executed, and the
lesson stuck.

## Retired providers

| Provider | What happened | Replacement |
|---|---|---|
| **Stooq** (was §7.3's primary OHLCV) | CSV endpoint serves a JavaScript proof-of-work challenge; bulk archive 401s | IB via the cache bridge |
| **CBOE** equity put/call | Every documented endpoint 403s | None. Costs a raw field, not the regime tag (§7.2 kept put/call out of the key) |

## GDELT collection is incomplete, and the window is closing

The API throttles aggressively — one request per 5 seconds, enforced with
sustained per-IP blocks. Collection is **intermittent, not banned**: one run
got four of fifteen symbols through before being refused.

**Collected (full 90 days):** SPY, MSFT, TSM, BRK.B
**Not collected:** AAPL, NVDA, AVGO, AMZN, GOOGL, META, JPM, XOM, UNH, JNJ, MRVL

```bash
uv run python scripts/gdelt_collect.py --days 90      # re-run opportunistically
uv run python scripts/gdelt_collect.py --status       # see what is missing
```

This is the **only** gap with an irreversible cost. GDELT serves a rolling ~3
months, so a day not collected is permanently absent from the Stage 8
evaluation (§7.5). Re-running picks up where it left off; it is idempotent.

## Stage 3 reasoning quality — residual issues

The vertical slice produces defensible decisions, found by reading ten of them
as the plan requires. Two things qwen3:8b still does that prompting has not
fully fixed:

- **It occasionally calls a momentum setup a "mean-reversion setup."** The
  decision is usually coherent anyway, but the label is wrong. Likely a
  genuine model limitation rather than a prompt gap.
- **It still infers a current position occasionally** ("the position is
  small"), despite the prompt stating that portfolio state is unknown. This
  resolves itself at Stage 6, when `compute_gaps()` supplies the real drift
  table — and §8 predicts exactly that: the drift table is *"the biggest
  reliability win"* because it turns an open-ended "how much should we buy"
  into a bounded "close this gap or don't".

Both were found by reading output, not by tests, and neither blocks the gate.

## Stage 4 residual

- **TSM has no `value` block.** Its share count is not under any concept name
  we try, and an ADR's share count differs from the ordinary shares anyway, so
  there is no market cap and therefore no yields. Honest rather than wrong —
  every one of those metrics states the reason. Fixing it means mapping ADR
  ratios, which is a real piece of work rather than another alias.
- **All four analysts frequently return `neutral`.** Sometimes correct, but it
  reads like hedging. §11's calibration plot is the thing that will actually
  answer whether their confidence means anything, and that is Stage 8.

## The hosted path is still unverified

`deep_hosted` is defined, `AnthropicClient` raises a clear unimplemented error,
and nothing routes to it — so Stage 5 runs fully local at $0. What remains:

- **Implement and verify `AnthropicClient`** against a real key. Deliberately
  not written blind: the two bugs in `scripts/fetch_bars.py` were both in code
  that had never executed.
- **Then run §11's `trader→deep_hosted` ablation** and the `all_hosted`
  preset, which is the actual answer to whether the one paid node earns its
  cost. The question is live, not settled.
- **Re-derive §12's cost table** against rates current at that time, as §12
  explicitly asks.

## Concept aliases: abandonment is handled, semantics are not

`_rows_for` now rejects an alias that is over 400 days behind a fresher one
(`ALIAS_ABANDONED_DAYS`), which fixed eight stale metric blocks. Two cases in
that sweep are only *arguably* right, because some alias lists are preference
chains over genuinely different measures rather than lists of synonyms:

- **`INTEREST` on a bank.** JPM stopped reporting `InterestExpense` in 2024-03
  and reports `InterestIncomeExpenseNet` now, so the fall-through takes it —
  but for a bank, gross interest expense and *net* interest income are
  different quantities, and `interest_coverage` wants the former. Current and
  arguably wrong beats two years stale and definitely wrong, so this is the
  better default; it is not a correct answer. The fix is a per-metric notion of
  which aliases are substitutable, which is more taxonomy modelling than
  Stage 5 warranted.
- **`SHARES` is deliberately left alone.** Shares outstanding vs
  weighted-average diluted sit 181 days apart on NVDA and JPM, below the
  threshold, so preference order still decides. That is the intended outcome —
  the threshold exists precisely so a quarter of reporting lag cannot redefine
  a metric — but it means the two concepts are never reconciled.

Worth revisiting if a fundamentals metric looks wrong on a financial. The
general lesson is recorded because it cost real time: **a ratio built from two
concepts can be wrong by a large factor and still render as a plausible
number**, and nothing downstream will reject it. NVDA reported a 1,914% gross
margin and the analyst reasoned about it rather than refusing it.

## Smaller things

- **`make gdelt-probe` has never run.** §7.5 flags a contradiction in GDELT's
  own docs (3-month rolling window vs an index back to 2017) and asks for it to
  be measured. The probe exists; the throttle has prevented it.
- **`deep_hosted` points at a placeholder model.** `config/models.yaml` names
  one; §12 asks for the cost table to be re-derived against current rates at
  Stage 5.
- **`confirmation.py` is uncarried.** Its renderer rewrite needs
  `FinalDecision`, so it lands at Stage 7. The original is in the ORB+GEX repo
  and at `git show 2005f11:confirmation.py`.
- **Sector ETFs have no fundamentals and no sentiment.** Expected — they are
  supporting series for relative strength only.
