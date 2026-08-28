# TradingAgents: migration and build plan

How to get from this repo to a working multi-agent research desk, in what order,
and what gets carried across.

Companion to [`tradingagents-architecture.md`](tradingagents-architecture.md).
That document is **why**; this one is **what, in what order, and what to copy**.

**Status:** plan agreed, build not started.

---

## Decisions taken

Four questions were open. All four are now settled, and several sections of the
architecture document were revised to match.

| # | Decision | Rules out |
|---|---|---|
| 1 | **Separate repository. Shared code arrives as copies.** | A `src/common/` shared package; any refactor of this repo |
| 2 | **The ORB+GEX engine is frozen and untouched.** | Extracting a shared execution layer as step zero |
| 3 | **Per-symbol metrics plus a shared macro regime tag.** | A peer-universe cross-section, a nightly universe-wide fetch, a local price store |
| 4 | **LangGraph from Stage 1, orchestration only.** | The plain `async def run_decision(...)`; also rules out `langchain-core` chat models, `ToolNode`, and `interrupt()` |

Decisions 1 and 2 together mean this is not really a *migration* in the sense of
moving code. It is a **new build that borrows four hard-won modules**. Naming it
honestly matters, because "migration" invites the instinct to keep the two
codebases in sync, and that instinct is exactly what decision 2 rejects.

---

## 0. The two-repo boundary

### What "frozen" means, precisely

The ORB+GEX engine works. It replays a session end to end, it places
human-confirmed paper orders, and its safety gate has been exercised. Nothing in
this plan is worth the regression risk of touching it.

So, concretely:

- **No commits to `src/trading_engine/` for the duration of this build.** Bug
  fixes to the ORB engine are of course fine; changes made *on behalf of* the
  new project are not.
- **No extraction, no shared package, no `pip install -e` across repos.** The
  four carried modules are copied, and the copies are then owned by the new
  repo and free to diverge.
- The one accepted cost: **a security-relevant fix to `assert_paper_account`
  must be applied twice**, and nothing will remind you. This is written into
  §15.11 of the architecture doc so it is at least on the record.

### One IB Gateway, two consumers

Both repos need IB. Running two Gateway containers means two logins to the same
IB account, which IB handles by evicting one of them. So: **one Gateway, owned
by this repo, joined by the new one.**

The Gateway service already declares a named network:

```yaml
networks:
  trading-network:
    driver: bridge
    name: trading-network
```

The new repo's compose file therefore declares it **external** and does not
define a Gateway service at all:

```yaml
networks:
  trading-network:
    external: true
    name: trading-network
```

Practical consequence: `make gateway-start` stays a command you run *in this
repo*. The new repo assumes a Gateway is already up and fails with a clear
message if not. Worth a `make check-gateway` target that says so in one line
rather than surfacing an `ib_async` connection timeout.

### Client ID allocation

`ib_async` connections with the same `clientId` do not error — **they silently
fail to connect**, which is among the least pleasant ways to lose an hour. The
ORB engine defaults to `client_id: 2`. Allocate a non-overlapping block and
write it down:

| Client ID | Owner |
|---|---|
| 2 | ORB+GEX engine (`config/orb-gamma-config.yaml`) |
| 3–9 | Reserved: additional ORB instances, one per symbol |
| **11** | **Research desk — `execute` process** |
| 12 | Research desk — ad-hoc scripts, `whatIf` probes, snapshot fetches |
| 13–19 | Reserved: research desk |

Note that **only `execute` connects to IB at all**. The `decide` process must
not import `ib_async` — that is the §0 split, and the cheapest way to enforce it
is a test that asserts the import is absent from the `decide` dependency tree.

### Ports

Unchanged and already correct: `ajj-ib-gateway:4004` from inside
`trading-network`, `127.0.0.1:4002` from the host. The new repo's `execute` runs
in a container on that network, so it uses **4004**.

---

## 1. The carry list

### Copied

| From this repo | Lines | Disposition |
|---|---|---|
| `execution/safety.py` | 71 | **Near-verbatim.** Keep `assert_paper_account` byte-identical if at all possible — it is the single most safety-critical function in either codebase and divergence is the risk. Two changes required: drop the `..models.DataMode` import, and re-point `assert_can_trade` at the new mode enum (`live` / `replay` / `replay_llm`), where `replay` and `replay_llm` both return `can_trade = False` |
| `execution/journal.py` | 45 | **Verbatim.** It is already generic — an append-only JSONL writer whose `write()` never raises. Only the event vocabulary changes |
| `logging_setup.py` | 73 | **Verbatim.** The benign-IB-code filter (10091 and friends) is needed the moment `execute` connects. Add the new noise sources — `httpx` at INFO is loud, and LangGraph/Langfuse both chatter |
| `execution/confirmation.py` | 113 | **Structure verbatim, renderer rewritten.** `CLIConfirmationGate`, `RejectAllGate`, the `Protocol` definitions, and above all the **type-the-ticker rule** and the `EOFError`/`KeyboardInterrupt` → decline behaviour all carry unchanged. `render_decision()` is rewritten for `FinalDecision` (shares) rather than `TradeDecision` (option bracket) |
| `tests/execution/test_safety.py` | — | Copy with its module. A carried safety function with no carried test is worse than not carrying it |

**`render_decision()` must gain three things** the options version had no
concept of, and they are the point of the whole exercise (§4 of the
architecture doc):

1. **`dissent`** — the preserved bear/safe case. If the confirmation prompt does
   not show you the argument against, the multi-agent structure bought nothing
   at the only moment it matters.
2. **`invalidation`** — "what would prove this wrong."
3. **An `expires_at` check**, refusing outright rather than prompting. A
   proposal generated after Tuesday's close must not be executable on Thursday.

It keeps `whatIfOrder` margin and commission in the prompt, exactly as now.

### Carried as pattern, not as code

| Idea | Where it lives now | Why it carries |
|---|---|---|
| Config validator that **rejects** `require_confirmation: false` at load time | `config.py:154` | The gate is not a setting. Same validator, new config model |
| Mode enum with behaviour on it (`can_trade`, `market_data_type`) rather than string comparisons at call sites | `models.py:16` | Puts the safety property on the type. `DecisionState.mode` should work the same way |
| Enforcing "cannot trade" **inside** the order path, not only at the call site | `safety.py:61`, `order_manager.py` | "so a future caller cannot route around it." Applies identically to `replay` and `replay_llm` |
| One conversion point into the internal `Bar` type | `bars/base.py::bar_from_ib` | The new repo has Stooq *and* IB *and* possibly yfinance producing daily bars. One normaliser, and record which source served each field (§7.4) |
| Nested config sections with a legacy-key folder | `config.py:101` | The pattern of making "which keys apply when" obvious in the file's shape |

### Explicitly not carried

| Module | Why not |
|---|---|
| `engine.py` — the ORB state machine | It is a five-stage intraday state machine. The new engine is a DAG that runs once per symbol per day. Nothing transfers |
| `strategy/opening_range.py`, `breakout.py`, `gex.py` | Intraday options logic. The new engine is daily equities |
| `bars/realtime.py`, `delayed.py`, `replay.py` | These stream intraday bars from IB. The new engine fetches daily history in one call at prefetch. The *replay concept* carries; the code does not |
| `execution/order_manager.py` | Option bracket construction. The new sizing path (§9) is share-based and driven by `PortfolioIntent` |
| `models.py` | Options-specific: `OptionRight`, `GEXResult`, `TradeDecision`. Only `DataMode`'s shape is worth imitating |

Roughly **300 of the 3,378 lines** in this repo carry across. That ratio is the
honest argument for decision 1: there was never enough shared surface to justify
a shared package.

### Infrastructure

| Artefact | Disposition |
|---|---|
| `docker/docker-compose-options-trader.yml` — Gateway service block | **Stays here, shared.** All those port and healthcheck comments are hard-won; do not re-derive them |
| Trader service block | Adapted into **two services**: `decide` (no IB, no `ib_async`) and `execute` (IB, no LLM) |
| `Makefile` | Copy the *conventions* — `make help`, `make config-check`, `make test-connection`. Not the targets |
| `example.env` | New file. The two-mode split (`TRADING_MODE` account vs `DATA_MODE` data) carries as a principle; the new variables differ |
| — | **New:** a Langfuse container for traces, and host-native Ollama per §10 |

---

## 2. Target layout

```
tradingagents/                      # new repo
  pyproject.toml                    # uv; LangGraph pinned exactly
  docker/
    docker-compose.yml              # decide + execute + langfuse; trading-network external
  config/
    models.yaml                     # §6 routing
    portfolio-intent.yaml           # §8
    providers.yaml                  # keys, rate limits, point-in-time flags
  prompts/                          # versioned; prompt_pack_version in state
  src/research_desk/
    graph/
      build.py                      # the LangGraph assembly -- the ONLY file importing langgraph
      nodes/                        # one module per node; framework-free bodies
    models/
      state.py                      # DecisionState + all §4 schemas
    llm/
      router.py                     # profiles -> provider clients
      structured.py                 # schema-constrained decode + one repair turn
    providers/
      registry.py                   # the single entry point (§2, tool-promotion path)
      stooq.py  ib.py  edgar.py  finnhub.py  finra.py  fred.py
      gdelt.py  cboe.py             # news tone/volume; put-call ratio
      cache.py  limiter.py
    metrics/
      indicators.py                 # §7.1 -- all arithmetic lives here
      regime.py                     # §7.2
      snapshot.py                   # build_market_snapshot()
    intent/
      engine.py                     # compute_gaps() drift table
      compliance.py                 # the Python veto
      sizing.py
    execution/                      # the ONLY package importing ib_async
      safety.py                     # <- copied
      confirmation.py               # <- copied, renderer rewritten
      journal.py                    # <- copied verbatim
      broker.py
    memory/
      store.py  reflect.py          # SQLite + numpy. No vector DB (§15.7)
    eval/
      replay.py  baselines.py  scoring.py
    logging_setup.py                # <- copied verbatim
  tests/
```

**One structural rule worth enforcing with a test:** `graph/build.py` is the only
module that imports `langgraph`, and `execution/` is the only package that
imports `ib_async`. Both are one-line `grep`-style tests, and both protect a
decision that is otherwise easy to erode a commit at a time.

---

## 3. Build stages

Sizes are **rough relative effort**, not schedule commitments: S ≈ a sitting,
M ≈ a few, L ≈ a week of evenings.

### Stage 0 — Skeleton and traces · S

Repo, `uv`, `src/` layout, pinned dependency set, compose file joining
`trading-network` as external, Langfuse container, host-native Ollama verified
per §10.

**Exit gate:** a two-node toy LangGraph runs and both nodes appear as spans in
Langfuse.

**Traps:** Ollama binds `127.0.0.1` by default and a container cannot reach it —
set `OLLAMA_HOST=0.0.0.0:11434`. Add a make target that curls `/api/tags` *from
inside the container*, per §10. Pin `langgraph`, `langgraph-checkpoint`,
`langgraph-prebuilt`, `langchain-core` to exact versions now, not later.

### Stage 1 — LLM router and structured output · M

`llm/router.py` + `llm/structured.py` + `config/models.yaml`. Schema-constrained
decode against Ollama's `format=<json_schema>`, validate, one repair turn with
the validation error appended, then `schema.degraded()` with `parse_failed=True`.

**Exit gate:** a smoke test asks a local 8B model for an `AnalystReport` and gets
a valid Pydantic object back — including when the first attempt is deliberately
made to fail schema validation.

**Traps:** the repair turn is the part that gets skipped and then desperately
needed at Stage 4. Build it now. Also start recording `LLMCallRecord` with the
**raw** response from the very first call — `replay_llm` is worthless
retroactively.

### Stage 2 — Providers, cache, metrics · L

Every provider behind the registry, every method taking `as_of`, every one
declaring `supports_point_in_time`. Then `indicators.py` computing the full
§7.1 set, `regime.py`, and `snapshot.py`.

**Exit gate:** `snapshot SYMBOL=SPY` prints a complete `MarketSnapshot` with
every §7.1 metric either populated or **explicitly `None` with a reason**, no
LLM involved. Then the same command for `AAPL` (has fundamentals) and for an
ETF (has none) — the ETF case is where a naive EDGAR path throws.

**Traps:** this is the largest stage and the one most likely to sprawl. EDGAR
needs a `User-Agent: Name email` header or it 403s, and 10 req/s or it 429s.
Stooq needs no key but has no support contract either — the fallback to IB must
be exercised, not just written. Build the cache before the second provider, not
after the sixth.

> **Do this on day one of Stage 2, before anything else in it:** stand up
> `providers/gdelt.py` and put a daily cron behind it, caching news tone and
> coverage volume for the intended universe. GDELT's open API serves only a
> **rolling three-month window** (§7.5), so every day not cached now is a day
> permanently absent from the Stage 8 evaluation, and no amount of later effort
> recovers it. It is one cron entry and a cache write, and it is the only task
> in this entire plan whose cost goes *up* the longer it is deferred.
>
> Same logic, lower stakes, for CBOE put/call — the archive goes back to 2006,
> so it can be backfilled whenever. Start GDELT now; CBOE can wait.

### Stage 3 — The vertical slice · M

`prefetch → market_analyst → trader → FinalDecision → proposal.json`. Three
nodes, one of them not an LLM.

**Exit gate:** **a sane, schema-valid decision end to end.** Read ten of them.
Not "does it parse" — *does a person who knows the symbol find the reasoning
defensible*.

**This is the highest-value de-risking step in the plan and it is worth
protecting from the temptation to build stage 4 first.** If market analyst →
trader does not produce something sane, more agents will not fix it; they will
produce a more confident version of the same nonsense.

### Stage 4 — Analysts and the research debate · L

The three remaining analysts, the four-way parallel fan-out, bull/bear
researchers, the facilitator, and all four termination conditions.

**Exit gate:** the debate stops for the right reason every time, and the
`stop_reason` is correct. Force each of the four conditions in a test.

**Traps:** the **reducer bug** — four parallel nodes appending to `llm_calls`
will silently overwrite each other without add-reducer annotations. Write the
test that runs the fan-out and asserts four records land (§4). The **novelty
guard** is not optional; local models restate rather than converge.

### Stage 5 — Risk debate and the fund manager · M

Risk trio, fund manager on `deep_hosted`, budget enforcement.

**Exit gate:** a full 14-node run completes inside
`RunBudget(max_llm_calls=20, max_wall_s=900, max_usd=0.50)`, and a deliberately
exhausted budget yields `HOLD` rather than a crash.

**Traps:** wire prompt caching on the repeated system prompts on day one of this
stage, not after the first bill.

### Stage 6 — Intent, sizing, compliance · M

`PortfolioIntent`, the four consumption channels, `compute_gaps()`, the sizing
minimum-of-four-caps, and `compliance.py`.

**Exit gate:** an `OrderPlan` is produced with **zero IB contact**, and a
deliberately non-compliant `FinalDecision` is blocked by Python regardless of
what the fund manager decided.

**Traps:** §8's rule that **analysts are intent-blind** is the one a future
refactor will "helpfully" fix. It needs the code comment the architecture doc
asks for, and ideally a test that asserts intent text is absent from analyst
prompts.

### Stage 7 — Execution · M

The `execute` process: copied safety, `whatIfOrder`, marketable limit orders,
the confirmation gate with dissent and invalidation.

**Exit gate:** **one paper trade, human-confirmed, with `assert_paper_account`
having fired twice** — once after connect, once immediately before
`placeOrder`.

**Traps:** paper accounts need their own credentials and must be *created* in
Client Portal first. `endDateTime` must be timezone-aware. Both are already
documented in §14 and both will still cost an hour.

### Stage 8 — Evaluation · L

Replay harness, baselines B0–B4, scoring, calibration plot.

**Exit gate:** **the B4 comparison.** If the full 14-node pipeline does not beat
a single LLM call by a margin outside B2's noise band, the multi-agent layer is
not earning its 14× cost.

**Traps:** report **residual return vs SPY**, not raw. Report the survivorship
caveat (§7.5) next to every number. And build the harness so switching A↔B4 is a
one-line config change, because a comparison that takes effort to re-run is a
comparison you will stop running.

### Stage 9 — Memory and reflection · M

SQLite + numpy dot product over reflection rows, keyed by `(symbol, regime)`.

**Exit gate:** reflections are retrieved *and measurably change decisions* —
which means an ablation with `memory.enabled=false`, not a log line showing a
retrieval happened.

---

## 4. Critical path, and where it is legitimate to stop

```
0 → 1 → 2 → 3 ──┬── 4 → 5 ──┬── 8   (go/no-go)
                │           │
                └── 6 → 7 ──┘
```

Stages 6 and 7 are independent of 4 and 5. If Stage 4 stalls on prompt quality —
likely — Stage 6 is productive work that needs no LLM at all.

**Three places it is rational to stop, and pre-committing to them is the point:**

1. **After Stage 3**, if the decisions are not sane. The failure is in the data
   or the prompts, and no amount of stages 4–9 addresses either.
2. **After Stage 8**, if A does not beat B4. Ship B4 — a single LLM call over
   the same excellent `MarketSnapshot`. Stages 0–3 and 6–8 all survive; only the
   debate machinery is discarded. **Most multi-agent systems fail this test**,
   and the plan is built so that failing it still leaves you with something
   good.
3. **Anywhere, if `decide` stops being fun to iterate on.** A 3–8 minute local
   run that you dread is a project that quietly ends. That is what Langfuse
   traces and node-level resume are for, and if they are not delivering, fix
   that before adding nodes.

---

## 5. The first week, concretely

1. Create the repo. Decide the name (see open items).
2. Pin the LangGraph set at exact versions and write a two-node toy graph.
3. Stand up Langfuse; confirm both node spans appear.
4. Verify host-native Ollama is reachable **from inside a container**.
5. Copy `journal.py` and `logging_setup.py` verbatim — free wins, and they make
   everything after them debuggable.
6. Copy `safety.py` and its test, re-pointing `assert_can_trade` at the new mode
   enum. Do this early even though IB is not touched until Stage 7: it is the
   module you least want to be writing in a hurry.
7. **Start the GDELT cache accumulating.** A standalone script and a cron entry,
   writing tone and volume for the intended universe into the cache directory.
   Not the provider layer, not wired to anything — just collecting. See the
   Stage 2 note for why this cannot wait.
8. Start Stage 1.

Do not touch the provider layer until the router returns a valid Pydantic
object. Stage 2 is the sprawl risk and it deserves a working LLM path to aim at.
**Item 7 is the single exception**, and it is an exception precisely because it
is not part of that layer — it is a cron job filling a directory, decoupled from
everything else, and its cost rises every day it is postponed.

---

## 6. Open items

| Item | Default if unresolved |
|---|---|
| **Repo name** | `tradingagents-desk`. `research-desk` is cleaner but ambiguous outside this context |
| **Which hosted model for `deep_hosted`** | Resolve at Stage 5, against rates current at that time. §12's cost table explicitly asks to be re-derived |
| **Universe** — *now needed in week one, not at Stage 6* | The `include` list in `portfolio-intent.yaml`. Start with ~10 liquid large-caps plus SPY; the per-symbol design (decision 3) makes a large universe expensive rather than impossible. **The GDELT cache is keyed by symbol, so whatever is not on the list from week one has no sentiment history at Stage 8.** Err wide — caching a symbol you never trade costs a few API calls a day; adding one later costs three months of unrecoverable history |
| **Company-name aliases for GDELT queries** | GDELT matches text, not tickers. `AAPL` needs to be queried as "Apple", and a naive query for `META` returns metaphysics. A hand-maintained alias dict per symbol, built alongside the universe. Get this wrong and the tone series is measuring something else entirely — worth eyeballing the returned articles for each symbol once |
| **Sector ETF mapping** for relative strength (§7.1) | A hand-maintained dict of ~11 SPDR sector ETFs. Free sector classification is poor; a dict for a 10-symbol universe is 10 lines and honest |
| **Whether `execute` gets a TUI** | No. CLI, per §9. Revisit only after Stage 8 |
| **Cron for `decide`** | Fine and intended (§15.8) — but not until after Stage 8. `execute` is never scheduled |
