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

## The conversational research agent — gated on Stage 8

**The goal.** A chat surface where the desk's agents research openly, propose
equities worth adding, and then *configure them into the desk*: universe entry,
provider symbols, sector ETF, `gdelt_query`, collection started.

**The gate is §11's B4 comparison, and it is not negotiable.** Build this only
once the full pipeline beats a single LLM call over the same snapshot by a
margin outside B2's noise band. If it does not, the plan says ship B4 and
discard the debate machinery — and a chat agent built first would be decorating
the part that got thrown away. *"Most multi-agent systems fail this test."*

**The architecture already permits it.** §2's "Keeping the door open to
tool-calling" is explicit that no-tool-calling is *"a statement about today's
models, not a permanent architectural commitment"*, and its three enabling
conditions are already satisfied: one registry (`providers/registry.py`),
tool-shaped signatures, and `as_of` on every method. The stated upgrade is to
generate schemas from the registry, hand them to a tool-capable node, and keep
the deterministic path as the fallback and the replay substrate.

**The hard constraint: it must be a separate process from `decide`.**
Tool-calling makes a run non-reproducible, and CLAUDE.md forbids it in that
path. `decide` stays deterministic; the chat agent is a sibling, like `execute`.

**What actually makes "auto-configure" safe is not the LLM.** The 2026-10-07
widening added 17 symbols by hand and every one of these had to be right:

- `gdelt_query` must be verified against real headlines, once, before the first
  collection — `collect_symbol` warns on a changed query because splicing two
  measurements into one history is invisible downstream. Measured: a plausible
  `("Visa Inc" OR "Visa card" OR "Visa payments")` returned Egyptian bank card
  launches and seven copies of a United Way community-funding story. It was
  measuring card marketing, not Visa Inc.
- `sector_etf` is hand-maintained because free sector classification is poor,
  and getting it wrong weakens `max_sector_pct` silently.
- `provider_symbols` differ per provider: BRK.B is `brk-b.us` / `BRK B` /
  `BRK-B` / `BRK-B`.
- §7.5: no survivorship-bias-free universe and no point-in-time index
  membership. Screening "today's best names" bakes in the bias §11 calls the
  single largest threat to its numbers.

So the agent should **propose a universe diff for human approval** — the same
"LLM proposes, Python vetoes" shape as `compliance.py` — and the mechanical half
belongs in a deterministic `desk universe add SYMBOL` helper it calls rather
than reimplements. That helper is the prerequisite and is worth having alone.

## Scheduling the GDELT collector — deferred to deployment prep

Run `make gdelt` by hand daily for now. Automating it belongs in a
post-validation deployment-prep task, alongside whatever else needs to run
unattended.

When it happens, **prefer launchd over cron on macOS.** A laptop asleep at 03:00
misses a crontab entry outright, and here a missed day is permanently absent
from the Stage 8 evaluation; launchd's `StartCalendarInterval` runs the job on
wake instead. It should call the collector through `uv` on the host rather than
`make gdelt`, which routes through the container and would need Docker running.

## Dead config in `universe.yaml`

Three per-symbol fields are declared and never read:

| Field | Status |
|---|---|
| `files_20f` | zero code references |
| `quality_metrics_apply` | zero code references |
| `has_fundamentals` | asserted only in `tests/test_config.py` |

The pipeline discovers absent fundamentals at runtime instead, from EDGAR
returning `None` — which `providers/edgar.py` is explicit is *"an answer, not a
failure"*. Harmless today, actively misleading to anyone adding a symbol by
hand, and directly in the way of automating that. Either wire them up or delete
them; leaving them is the worst of the three.

## GDELT: what is left, as of 2026-10-07

The universe widened to 32 collected symbols and the API put this IP in a
sustained throttle partway through. Three things remain, in this order.

**1. Verify two queries BEFORE their first collection.** `TEM` and `AMD` are
the only new symbols whose `gdelt_query` was never checked with `--inspect`:

```bash
uv run python scripts/gdelt_collect.py --inspect TEM
uv run python scripts/gdelt_collect.py --inspect AMD
```

Order matters and is not a preference. `collect_symbol` warns on a changed
query because the query *defines what the series measures*, and splicing two
measurements into one history is invisible downstream. Fix first, collect
second. `universe.yaml` carries the same warning inline at the entries.

This is not hypothetical: the first `V` query was
`("Visa Inc" OR "Visa card" OR "Visa payments")`, which looked careful and
returned Egyptian bank card launches, a World Cup economy story, and seven
copies of a United Way community-funding piece. It was measuring card
marketing. `CAT` was checked and is fine.

**2. Backfill the 28 symbols with no history.**

```bash
uv run python scripts/gdelt_collect.py --days 90
uv run python scripts/gdelt_collect.py --status     # expect 32/32, no gaps
```

Two requests per symbol covering the whole window in one call, so ~64 requests
at one per five seconds -- **5-6 minutes**, not hours. The constraint is the
throttle, not the volume: a sustained per-IP block survives the full backoff,
and the collector says so and stops rather than extending it. Re-run
opportunistically; it is idempotent and resumes.

**3. Run `--probe-window`, which has still never run.**

```bash
uv run python scripts/gdelt_collect.py --probe-window
```

§7.5 flags a contradiction in GDELT's own documentation -- the API is described
as searching "a rolling window of the last 3 months" while also referencing an
index back to 1 January 2017 -- and asks explicitly for the real reachable
`startdatetime` to be **measured** rather than trusted.

Worth doing before scoping the conversational research agent above, because it
decides one of that design's load-bearing constraints:

* if the window really is ~3 months, a day not collected is gone and late
  universe additions permanently cost history;
* if more is reachable, late additions are backfillable and the strongest
  objection to an agent that discovers symbols disappears.

Write the answer and its date into §7.5, replacing the caveat.

## Stage 7 residual

### The exit gate still needs one human-placed trade

Everything up to the gate is verified against the live paper account
(`DUT052405`): connect, both `assert_paper_account` firings, reconciliation,
a delayed bid/ask quote, the marketable limit, a real `whatIfOrder` (init margin
9,845.94, commission 1.00 USD), and the full prompt. `execute --dry-run`
declines and leaves no receipt.

What remains is the part that cannot be automated by design -- typing the ticker:

```bash
desk decide --symbol TSM     # or whatever has headroom
execute                      # then type the symbol
```

Then confirm in the log that the account was verified twice and that a parent
LMT plus a child STP were both accepted.

### The stop covers only the shares just bought

`broker.build_orders` attaches a protective `STP` to every BUY, which is what
makes sizing's `cap_risk` true rather than notional -- `per_trade_risk_pct /
stop_dist` sizes every position on the arithmetic that the stop bounds the loss.

**It protects the new shares only.** Add to a symbol that already has a stop and
the account ends up with two stops rather than one for the whole position.
Proper stop management means cancel-and-replace on every add, which needs order
state the desk does not currently track, and is a real piece of work rather than
another flag. Until it exists, adding to a held position leaves a stop ladder
rather than a single protective level.

### The book: not committed, and now in `data/` rather than `config/`

It is state, and from Stage 7 the broker owns it. It held an account balance in
version control, and the one reason it was committed -- "so the Stage 6 gate is
reproducible from a clone" -- stopped being true once the gate started building
its own book from `scripts/seed_portfolio.py:SEED_BOOK` plus the bars cache.

The suite now passes with the seeded book, the real book, or none. That
separation was forced by the first real `execute` run: writing the account's
actual (all-cash) book over the seed turned twelve Stage 6 tests red for reasons
that had nothing to do with what they tested -- the third time in two days that
a test depended on a file that moves.

### The seeded book is fiction, and reconciliation says so

The committed seed was $250,000 across fifteen positions. The real paper account
is **$1,003,109.96 in cash with no positions at all**, so every proposal sized
against the seed is refused by reconciliation -- correctly. Position sizes are
roughly 4x larger against the real book, and with fifteen free slots almost
everything is buyable.

`make portfolio-seed` still installs the demo book, which is useful for working
on `decide` without a Gateway. Just expect `execute` to refuse it.

## Langfuse traces nodes, not model calls

**What works.** `obs/tracing.py:traced_node` wraps every node body at graph
assembly time, so each node is one span. Verified 2026-10-08: 354 spans in
project `desk`, continuous from 2026-08-28 (Stage 0) to the previous night's
runs, covering all 17 node types -- `prefetch`, the four analysts, bull/bear/
facilitator, `trader`, the risk trio, `fund_manager`, `compliance`, `persist`.

**What is missing, and it is the thing Langfuse is actually for.** There are no
**generations**: no prompt, no completion, no token counts, no model name, no
cost, no per-attempt detail. `_summarise` deliberately sends only scalar fields
and collection *lengths*, for a stated reason -- *"shipping all of it to the
collector on every node makes traces unreadable and slow"* -- and that reasoning
holds for the node INPUT. It is the wrong trade for the model call itself.

So Langfuse answers "the bull researcher ran and took 30s" but not "what did it
say", and it cannot answer "what did this cost" or "how many tokens" at all.

**The data is not lost.** `LLMCallRecord` in the decision journal already carries
everything a generation needs: `raw_response`, `thinking`, `prompt_tokens`,
`completion_tokens`, `latency_ms`, `model`, `model_digest`, `usd`, `attempt`,
`parse_failed`. `desk review --full` prints the summary of it. Nothing is
unrecoverable; it is simply not in the place built for looking at it.

**Why this matters before Stage 8 rather than after.** §11 wants a calibration
plot and §12 wants the cost table re-derived. Both are computable from the JSONL,
but the per-call view is also how you debug *why* a weak local model produced a
bad report -- which is most of Stage 4-5's remaining work. Doing it by grepping
JSONL when a UI exists for it is the kind of friction §4's third stopping rule
warns about: *"if `decide` stops being fun to iterate on"*.

**Sketch of the fix.** `llm/structured.py` already returns `LLMCallRecord`s, and
`traced_node` already holds an open span when they are produced. Emit a child
observation per record with `as_type="generation"`, mapping `prompt_tokens` /
`completion_tokens` to Langfuse's usage fields and `model` to its model field so
its own cost table applies. The repair turn should be its own generation
(`attempt=2`), because a run where every call needed a repair is a different
run and the trace should show that.

Keep the existing node spans: the parent/child shape is what makes a 14-node
run readable.

## Langfuse: one API key pair, one project

Measured, because it decides what is safe to change. `api_keys` has exactly one
row, scoped to project `desk`, and `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` /
`SECRET_KEY` in `docker/docker-compose.langfuse.yml` are the **same variables
the app reads** (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`).

Consequences worth writing down:

* **The `LANGFUSE_INIT_*` block is a bootstrap, not a project manager.** It
  cannot give you two live projects. Pointing it at a new `PROJECT_ID` without
  also issuing a new key pair tries to provision a project with keys already
  bound to `desk`.
* **Changing the project means changing where `decide` writes**, because the
  app and the provisioner share the key variables. That is fine and even useful
  for a clean project per experiment (Stage 8 A/B), but it is an either/or
  rather than additive.
* **A project created in the UI has no API key until you generate one there.**
  That is why the hand-made `trading-llm-test-test` project was empty: nothing
  could ever have written to it.

**What breaks if the wrong variable changes** -- none of these destroy trace
history, which lives in ClickHouse keyed on `project_id`:

| Variable | What actually breaks |
|---|---|
| `LANGFUSE_SALT` | API keys are hashed with it, so the SDK stops authenticating until new keys are issued. History intact. |
| `LANGFUSE_ENCRYPTION_KEY` | Secrets stored *inside* Langfuse (LLM keys pasted into the UI, integration creds) stop decrypting. History intact. |
| `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` | New events go to whatever project the new pair belongs to. Old events stay where they were. |
| `NEXTAUTH_SECRET` | Browser sessions only. Log in again. |

**And the one that actually caused trouble:** `LANGFUSE_INIT_USER_EMAIL` and
`LANGFUSE_INIT_USER_PASSWORD` were empty, so first boot provisioned the org,
the project and the API key but **no user** -- an org nobody is a member of.
Tracing worked perfectly and was invisible in the UI for six weeks. Setting
those two and recreating `langfuse-web` is additive and safe.

## The book moved to `data/`, and the read-only mount is why

Stage 7's first real fill could not be recorded:

    OSError: [Errno 30] Read-only file system: '/app/config/portfolio.yaml'

`execute` mounts `../config:/app/config:**ro**`, and that is **correct and
should stay**: the process that places orders must never be able to rewrite
`portfolio-intent.yaml` and relax its own risk limits. If it could, the two-key
design collapses -- the whole point is that Python's veto is not negotiable by
the thing being vetoed.

So the mount was right and the book was in the wrong directory. `config/` is
policy that Python enforces and `execute` must not touch. `data/` is state:
gitignored, mounted read-write, and already home to the journal, the proposals
and the receipts. The book belongs with those, and `intent/engine.py`'s
`portfolio_path()` is the single place that decides.

This is the same lesson as the day before, one level deeper. Then: the book
should not be *committed*, because it is state rather than configuration. Now:
it should not be *in config/* either, for exactly the same reason. The first fix
addressed the symptom and left the cause.

## `portfolio-refresh` cannot see a new position, and the advice said otherwise

`--refresh` re-prices positions the book **already lists**, from the bars cache,
and never talks to IB -- the script says so twice ("reads the cache directly and
never constructs a provider, so it cannot fetch"). After a fill the book still
said zero positions, and refresh had no way to discover otherwise.

`execute` nevertheless printed *"The book is now out of date -- run `make
portfolio-refresh` once the fill settles"*, which pointed at the one command
that could not do it. Fixed, and the gap it was papering over is now a real
command:

    make sync-book          # quantities and cash, FROM THE BROKER
    make portfolio-refresh  # then mark those positions to market

They compose deliberately: `sync-book` is authoritative about *what you hold*,
`portfolio-refresh` about *what it is worth*. Marks from `sync-book` alone are
each position's average cost, because fetching fifteen live quotes to write a
file would be slow, rate-limited, and no more accurate than the next refresh.

## `make check-gateway` was overstating what it proved

It reported "desk-ib-gateway reachable" on the strength of a TCP connect to
4012. §14 already documented why that is not evidence: the image runs socat
relaying 4004 -> 4002, and **socat listens from container start whether or not
the Gateway ever logged in**. So the probe says healthy while the Gateway sits
on the login screen.

This is not theoretical. One IB username supports one session, so logging into
IB's web portal with the same credentials can evict the Gateway's -- and nothing
about the port changes when it does. The symptom is a web UI offering to "log
out of the other session" while every local check still looks green.

`check-gateway` now says what it actually tested ("port answers (socat)") and
names the deeper check. `make gateway-session` does a real `managedAccounts()`
round-trip and prints the account, the positions and NetLiquidation. It lives in
`execute` because that is the only process permitted to import `ib_async`, which
is also why `desk doctor` keeps its bare TCP connect.

## Sweeps accumulate expired proposals

A full `make decide-all` writes 31 proposals, and `cadence.proposal_ttl_hours`
is 18 -- so by the next morning every one of them is expired. That is the TTL
doing its job and is what makes "sweep today, trade tomorrow" safe: tomorrow's
trades come from tomorrow's sweep, and today's thinking cannot be executed by
accident once it is stale.

The side effect is that `data/proposals/` grows by ~31 files a day and
`execute --list` grows a long EXPIRED section. Nothing breaks -- `assert_actionable`
refuses an expired proposal before touching the network -- but the listing stops
being scannable.

**Follow-up: `make prune-proposals`.** Delete proposals that are expired AND
have no receipt, keeping anything that was acted on. Receipts are the audit
trail and must survive; an expired never-executed proposal is just noise. Worth
a `--keep-days N` so a week of history stays browsable.

Deliberately not built yet: it deletes files, and the right retention policy is
clearer after a few weeks of real sweeps than it is now.

## The batch executor, as built

`execute --all` exists. It reviews every actionable pending proposal in one
screen, then walks them individually, each with its own typed ticker. What it
adds over running `make execute --symbol X` N times is a **running book**: each
order is re-checked against the account as it is after the previous fill.

That was not a convenience. Five of `rules.check`'s rules read
`post_trade(portfolio, ...)` — `max_positions`, `min_cash_pct`,
`max_gross_exposure_pct`, `max_sector_pct`, `max_position_pct` — and `_run`
loaded the book once and never advanced it. A shell loop would have walked past
all five. The manual `make sync-book` between orders was hiding it by accident.

Three decisions worth keeping, because each had a plausible alternative:

**The gate stayed per-order.** The temptation was one approval for the whole
batch, and the objection to "type the ticker fifteen times" is real — it is a
gate people learn to defeat. But one token standing for fifteen orders makes
the ticker mean *a set* rather than *a trade*, and the ticker is the whole
mechanism. What makes N prompts tolerable instead is that each one carries new
information: the live limit price and the whatIf margin, neither of which
exists until that order is priced. The batch review screen carries the
portfolio-level decision, so the per-order prompt is only confirming this
order at this price.

**An ambiguous fill stops the batch.** `wait_for_terminal` waits on the parent
only — the child stop is GTC and is supposed to stay working — and a timeout is
not an error. The order may fill a second later; what the process cannot do is
describe the book well enough to check the next order against it. Same rule as
reconciliation's: a thing we cannot describe is a refusal, not a warning. The
unreached proposals keep no receipt and stay pending.

**It writes the book, and single-order `execute` still does not.** A batch that
places five orders and leaves `data/portfolio.yaml` five orders stale is a trap
for the next `decide`. What makes the write safe is that it reconciles first:
unlike `--sync-book`, which overwrites unconditionally, this confirms the delta
is the one the batch intended and refuses if it is not.

Still open, and deliberately:

* **Order priority.** Orders go in the sequence shown on the review screen
  (newest proposal first), so when a cap binds the earlier ones take the
  headroom. Sorting by conviction is defensible and is cleverness in the wrong
  file until there is a reason for it.
* **Stop cancel-and-replace** (below) is still unbuilt. Dedupe-by-symbol keeps
  a single batch from creating the double-stop case, but adding to an existing
  position across two days still leaves two stops.

## The 31 historical degraded runs are not being retro-fixed

`NodeError.severity` split "an input was absent" from "the machinery broke"
(architecture §5, "As built at Stage 7"). Before it, 31 of the 32 degraded runs
in the journal were degraded only because an analyst had no data, and each one
rewrote an approved BUY to a HOLD.

Those runs stay as they are. The JSONL is §1's source of truth for replay and
audit — it records what the code *did*, not what today's code would do — and
editing it so the HOLDs become BUYs would make every one of them a decision
nobody ever made and no human was ever shown. Replay would then be a lie about
the one thing it exists to prove.

The practical consequence is that **any analysis spanning the split has to read
`FinalDecision.absent_analysts` as well as `degraded`**, and that Stage 8's hit
rate cannot be computed across the boundary without saying which side a run came
from. The run ids are dated, so the boundary is findable; this commit is the
line.

Also: those 31 HOLDs are not just mislabelled, they are *missing decisions* —
the fund manager's BUY was never acted on and never will be. They are usable as
evidence about the pipeline, not as evidence about the strategy.
