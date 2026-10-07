# Known gaps and follow-ups

Deliberate omissions, with what unblocks each. Nothing here is a stub pretending
to work: every provider that exists is verified against live data, and every
metric that cannot be computed says why in the snapshot itself.

## Blocked on a free API key

| What | Needs | Consequence today |
|---|---|---|
| **Finnhub** — company news, earnings calendar, EPS surprises, recommendation trends | `FINNHUB_API_KEY` (free, 60/min) | No events block. The earnings blackout now runs on an **estimate** from EDGAR filing cadence and therefore WARNS instead of blocking — see "The earnings blackout warns" below |
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

## Stage 6 residual

### The earnings blackout warns rather than blocks, deliberately

`risk.earnings_blackout_days_before/after` is the one compliance rule whose
input is not a known number. There is no `FINNHUB_API_KEY`, and EDGAR publishes
no forward calendar, so `intent/earnings.py` projects the next report date from
the **cadence of past 10-Q/10-K filings** — which the cache already holds for
the universe, so it needs no new artifact and nothing to maintain.

A median-of-intervals estimate is good to roughly one to two weeks. Blocking on
it would veto about a month of every quarter on a guess, *and* would appear in
the log as a real earnings veto. So:

| Input | Severity |
|---|---|
| a **confirmed** date (an earnings calendar, when one exists) | `block` |
| an **estimated** date inside the window | `warn`, surfaced in the confirmation prompt |
| **no usable history** (an ETF, a foreign private issuer) | `warn`, saying it could not be evaluated |

Every other rule in `compliance.py` blocks, because every other rule reads a
number that is actually known. **Flipping this one to `block` is a single field
and no schema change** once a calendar is wired up — `Violation.severity`
already carries it and `tests/intent/test_compliance.py` already asserts the
confirmed path blocks.

Known limits of the estimate, all of them visible in the output:

- **TSM returns `unknown`.** A foreign private issuer files a 20-F annually and
  announces quarterly results on 6-K, which it also uses for press releases and
  dividend notices — 664 of them against 13 20-Fs in this cache. Including 6-K
  would put the median interval at a few days and produce a confident date that
  is nonsense, so it is excluded and the symbol honestly reports that the check
  could not be evaluated.
- **XOM falls back to a nominal 91-day quarter.** EDGAR's `recent` submissions
  index is capped, so a prolific filer's periodic reports get pushed out of it
  and only one 10-Q is visible. The estimate says so in its own `source` string.

### `conviction_w` was removed — resolved before Stage 7

§9 specified `conviction_w = target_weight_pct * conviction` as the first of the
four sizing caps, and Stage 6 shipped it literally. The first live run showed
the cost: the drift table offered TSM 5.83%, the trader proposed 5.83% at 0.75,
the fund manager **adjusted down to 4.5%** for overbought risk at 0.65, and
sizing multiplied to 2.93% — **three shares.** The reduction for risk landed
twice, once as a judgement and once by the formula.

**The reason it was removed is not the double-damping, which is only the
symptom. It is that `conviction` already has a job and sizing is not it.**
`prompts/trader.md` promises the model that conviction *"will be measured
against realised outcomes"*, and §11 makes the calibration plot (realized hit
rate bucketed by stated confidence) *"cheap and usually the most damning
diagnostic"*. A number under measurement must not also be a control input: once
it sets position size there is pressure to inflate it, and the plot then
measures a gamed quantity. §11's go/no-go rests on that plot.

As built, still four caps and none duplicating another:

```
target_w = min(requested, cap_intent, cap_position, cap_risk)
```

The same TSM decision now sizes to **+11 shares**, bound by `requested` — the
4.5% the fund manager actually approved. `conviction` has **zero numeric
consumers** anywhere in the codebase, and the guard is behavioural rather than a
name check: `tests/intent/test_sizing.py` sizes one decision at conviction 0.01
through 1.0 and asserts the share count never moves.

Both prompts were updated to close the loop at the source — the trader is told
plainly that conviction does not affect size, and the fund manager is told that
lowering conviction and lowering target weight are two separate decisions, so
cutting only conviction approves the original size with a caveat attached.

**Theme-level `conviction:` stays prose-only**, by decision. It was never read
numerically, despite a code comment in `models/intent.py` claiming it fed
`conviction_w` (now corrected). As a size multiplier it would be redundant with
`target_weight_pct`: if the desk wants 28% in a theme the target should read
28%, not 35% scaled by 0.8, which leaves two numbers that can disagree about one
intention.

### A stale prompt section, found while doing the above

`prompts/trader.md` still carried a `WHAT YOU DO NOT KNOW` section reading *"You
have not been told what the desk currently holds. Portfolio state and the drift
table arrive at a later stage."* True at Stage 5, and flatly contradicted at
Stage 6 by the drift table sitting in the same prompt. It now describes what the
trader does have (current weight, target, band, gap in percent and dollars, free
slots) and what it still does not (fill price, whether it fills at all, the other
symbols' theses).

Worth noting because of what it implies about the earlier runs: FOLLOWUPS
previously recorded the trader *"still infers a current position occasionally,
despite the prompt stating that portfolio state is unknown"* — and the prompt was
still stating exactly that while handing it the real numbers. Some of the timid
sizing attributed to `conviction_w` may have had this as a second cause.

### `max_gross_exposure_pct` is inert while the book is long-only

`min_cash_pct: 10` and `max_gross_exposure_pct: 95` sum to 105. In a long-only
unlevered book `cash_pct + gross_pct == 100` identically, so the cash floor
already caps gross at 90 and the 95 can never bind. That is redundancy, not a
contradiction — no order is blocked by both — and the 95 starts mattering the
moment `universe.allow_shorts` becomes true, since gross counts both legs.

`RiskLimits.effective_gross_cap_pct()` resolves which one actually binds, and
the violation message names it, so the log does not blame a limit that was not
the real constraint.

### The book is a seed, not a real account

`config/portfolio.yaml` is committed so the Stage 6 gate is reproducible from a
clone. Its share counts are hand-chosen to make every rule reachable — the book
sits at `max_positions`, two symbols sit at `max_position_pct`, one theme is
~17pp underweight and another ~5pp overweight, and TSM is far enough below its
target that a BUY actually sizes. Every *price* is a real cached close.

At Stage 7 `execute` overwrites this file from the live paper account and
`source:` becomes `ib`. Until then `make portfolio-refresh` re-marks it from the
bars cache, and `as_of` tracks the **oldest** mark it used so that re-marking
against a stale cache cannot reset the staleness clock.
