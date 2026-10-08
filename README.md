# Research desk

A multi-agent LLM research desk for daily equities and ETFs, modelled on the
TradingAgents paper (arXiv 2412.20138). Local models by default, hosted for the
one node where reasoning quality moves the outcome. **Paper account only, and
every order requires human confirmation.**

Two documents carry the reasoning, and they are worth reading before the code:

- [`tradingagents-architecture.md`](tradingagents-architecture.md) — *why*.
- [`tradingagents-migration-plan.md`](tradingagents-migration-plan.md) — *in what
  order*, and what was carried across from the ORB+GEX engine.

## The shape of it

Two processes that communicate through a file, never a shared event loop:

```
[ decide ]  agents + LLM + free data  →  proposal.json  →  [ execute ]  human gate + ib_async
 no IB connection, no ib_async import                      no LLM, no network beyond IB
```

## Build status

| Stage | Deliverable | State |
|---|---|---|
| **0** | Skeleton, pinned deps, compose, Langfuse, toy graph | **done** |
| **1** | LLM router + structured output | **done** |
| 2 | Providers, cache, metrics | **2a-2c done** (76 metrics); Finnhub + FRED need keys |
| **3** | Vertical slice → `proposal.json` | **done** |
| **4** | Analysts + research debate | **done** |
| **5** | Risk debate + fund manager | **done** (local judge; hosted deferred) |
| **6** | Intent, sizing, compliance | **done** (earnings blackout warns on an estimate — see gaps) |
| **7** | Execution + confirmation gate | **done** (one paper trade still to be placed by hand — see gaps) |
| 8 | Evaluation, B0–B4, the go/no-go | not started |
| 9 | Memory + reflection | not started |

## Getting started

```bash
make setup
make up          # Langfuse; Ollama stays native, see below
make shell       # a bash shell inside the desk container -- start here
```

**Containers are the default.** `make shell` drops you into the `dev` service
with this repo bind-mounted at `/app`, so edits take effect with no rebuild.
Inside, `desk`, `pytest` and the scripts are all on `PATH`:

```
desk doctor
desk snapshot --symbol AAPL
pytest -q
```

Every `make` target runs in that container too. Append `HOST=1` to run on the
host instead, which is useful when Docker itself is what's broken:

```bash
make test            # in the container
make test HOST=1     # on the host, via uv
```

The two contexts reach some services at different addresses — Ollama,
Langfuse, the IB Gateway — and the compose file plus
`config.resolve_ib_endpoint()` handle that, so the commands are identical
either way. `make verify` prints which context it ran in.

**Ollama is the one deliberate exception and stays native.** Architecture §10:
containerised on macOS there is no Metal passthrough, so it is CPU-only in a VM
— measured 5–15× slower, which turns a 4-minute pipeline into 40+. Containers
reach the host daemon via `host.docker.internal`, and on current Docker Desktop
that works even with Ollama bound to `127.0.0.1` (a correction to §10 — see
there). On Linux you do need `OLLAMA_HOST=0.0.0.0:11434`.

Then, after any pull, the one command that checks everything:

```bash
make verify
```

It runs every exit gate in order — config, tests, environment, toy graph,
Stage 1, Stage 2, Stage 6, Stage 3 — and stops at the first real failure.
Stage 6 runs *before* Stage 3 deliberately: it needs no model and no network, so
it takes a second and tells you the book is stale before you spend three minutes
of inference sizing against it. `make doctor` alone gives just the environment
report.

`make help` lists everything. The three checks worth knowing:

- `make check-ollama` — curls `/api/tags` **from inside the container**, which is
  where the `OLLAMA_HOST=127.0.0.1` trap actually bites (architecture §10).
- `make check-gateway` — whether **our** Gateway is up, and whether the ORB+GEX
  engine's is conflicting with it. See the warning below.
- `make models` — pulls every Ollama model `config/models.yaml` names. `doctor`
  warns with the exact `ollama pull` command when one is missing, rather than
  letting Stage 1 die on a 404 that looks like a router bug.
- `make toy-graph` — the Stage 0 exit gate: a two-node LangGraph run whose nodes
  both appear as spans in Langfuse.
- `make bars` — fetch daily bars from IB into the cache (needs the Gateway).
  `make data` shows what is cached, from which source, and how stale.
- `make snapshot` — the Stage 2 exit gate: all 76 metrics for a symbol, each
  either populated or explicitly unavailable *with a reason*. No LLM.
  Needs `SEC_EDGAR_USER_AGENT` set, or EDGAR 403s without saying why.
- `make gdelt` — collect news tone. **Its window closes**: GDELT serves a
  rolling ~3 months, so uncollected days are unrecoverable. Run
  `--days 90` once to backfill.
- `make decide` — the Stage 3 exit gate: a real decision end to end,
  written to `data/proposals/`. `SYMBOL=NVDA make decide` for another name.
  Since Stage 6 the output also carries the `OrderPlan` — the share count, the
  four caps and which one bound, and any compliance violation.
- `make decide-many SYMBOLS="TSM AMD"` / `make decide-all` — a **research
  sweep**: `desk decide` once per symbol, independently, in one container.
  `decide-all` is the whole 31-symbol universe at ~100s each, so **~50
  minutes**. Safe because each run reads the book and never writes it. Past the
  third actionable symbol the cadence limit vetoes the rest to HOLD — that is
  the limit working, and `desk review --wanted` shows what the pipeline
  actually decided.
- `make review` — read past runs out of the decision journal: the decision, the
  order plan, the debate, the violations. `SYMBOL=TSM make review`, or
  `desk review --last 10` for a listing and `--full` for every analyst and the
  raw model calls. The journal is the source of truth for replay (§1), and this
  is what reads it.
- `make portfolio` — the Stage 6 exit gate: the book and the drift table, with
  **no model and no broker**. Exits non-zero if the marks are stale, because
  sizing against last week's weights is how a position gets doubled.
- `make candidates` — channel 1 of §8: universe minus exclusions minus earnings
  blackout minus at-max-positions, computed before any model runs. `--earnings`
  also prints every symbol's next-report estimate.
- `make portfolio-seed` / `make portfolio-refresh` — the only two things that
  write `data/portfolio.yaml`. Both read the bars cache and **cannot fetch**,
  so they run with the Gateway down. Do not hand-edit the file.
- `make execute` / `execute --list` — **the Stage 7 gate.** Reads a
  `proposal.json`, reconciles the book against the broker, prices a marketable
  limit, runs `whatIfOrder`, shows you the dissent and the invalidation, and
  places nothing until you type the ticker. `execute --dry-run` walks the whole
  path and declines. **Never scheduled** (§15.8).
- `make test-stage6` — the other half of the Stage 6 gate: an `OrderPlan`
  produced with zero IB contact (asserted by making every socket connection
  fail), and a deliberately non-compliant decision blocked regardless of what
  the fund manager decided.
- `make smoke` — the Stage 1 exit gate: a real `AnalystReport` out of a local 8B
  model, showing attempts, latency, tokens and the model digest.

The live model tests are deselected from `make test` (they add ~30s and need
Ollama up); `make test-smoke` runs them explicitly, and `make verify` includes
them.

`make test-smoke` sets `REQUIRE_SMOKE=1`, which turns "Ollama unavailable" from
a skip into a failure. Without it the gate exited 0 on four skips — green
without having loaded any weights. That is also how a container-only bug stayed
hidden: the test resolved Ollama with the bare `Settings()` default of
`127.0.0.1`, which is correct on the host and points at the container itself
inside `dev`. Tests that reach a real service use `load_settings()`; the
offline ones keep `Settings()` deliberately, so a local `.env` cannot change a
result.

## The two processes, and why they share nothing but files

```
[ desk decide ]                          [ execute ]
 agents, LLM, free data                   human gate + ib_async
 no IB connection                         no model, ever
 no ib_async in its import tree    -->    proposal.json    -->    one order
```

`decide` has never held a broker connection and `execute` has never called a
model. That is what keeps the dangerous half small enough to read in one
sitting, and `tests/test_layering.py` fails the build if either leaks into the
other.

**Three safety properties, each enforced rather than intended:**

1. `assert_paper_account()` fires **twice** — once after connect, once
   immediately *before* `placeOrder`. The second catches a reconnect that
   landed on a different account; the port number is not a guarantee.
   `tests/execution/test_broker.py` asserts the call *sequence*, so a check
   moved after the order fails the build.
2. **A mismatch between the book and the account is a refusal, not a warning.**
   The share count was computed from `data/portfolio.yaml`, so if the file is
   wrong the number is wrong and there is nothing safe to approve. You get the
   per-symbol diff and an offer to rewrite the book from the broker.
3. **Approval requires typing the ticker**, a closed stdin declines, and an
   expired proposal refuses before a prompt is drawn. There is no flag that
   turns any of it off.
4. **Compliance is re-run at execute time**, against the book as it is then.
   The share count was sized at *decide* time, and reconciliation only proves
   the book matches the broker — not that the order still passes the vetoes.
   Those have the same answer only when nothing changed in between, which
   batching and "decided this morning, executed after lunch" both break.

## Where a decision's artifacts land

Three places, answering different questions:

| Question | Where |
|---|---|
| What did it decide, and what order? | `data/proposals/*.json` — ~2KB, pretty-printed, just open one |
| Why? What did each agent say? | **Langfuse at http://localhost:3000** — per-node spans with the real prompt and response. Start here. |
| Everything, replayably | `data/journal/decisions_YYYYMMDD.jsonl` — the full `DecisionState` per run, including `llm_calls[].raw_response` |

The journal is ~50KB per run, so `desk review` exists to read it rather than
`cat`. `proposal.json` is what `execute` consumes and the only thing it reads.

## Never run both IB Gateways at once

This repo runs **its own** Gateway, `desk-ib-gateway`, started with `make
gateway-start`. The ORB+GEX engine runs `ajj-ib-gateway`. They use the **same IB
credentials**, and one IB username supports exactly one Gateway session — so
starting the second one means IB evicts somebody, possibly the one holding
orders.

Two guards, and neither is decoration:

- `make gateway-start` runs `scripts/check_gateway_exclusive.py` first and
  refuses if the other Gateway is up, naming what it found and how to stop it.
- The container defaults to `EXISTING_SESSION_DETECTED_ACTION=secondary`, so it
  steps aside rather than taking the session over. The ORB engine uses
  `primary`. Overriding ours to `primary` will kill a running ORB session.

Host ports are deliberately different so a connection is never ambiguous:

| | Host API | Host VNC |
|---|---|---|
| ORB+GEX `ajj-ib-gateway` | 4002 | 5900 |
| Research desk `desk-ib-gateway` | **4012** | **5912** |

**The exclusivity guard is the one thing here that always runs on the host, and
it is the exception to "containers are the default."** Both of its checks read
host state: `docker ps` needs the Docker CLI and a mounted socket, which the
`dev` service deliberately does not have, and `127.0.0.1:4002` inside a
container is the *container's* loopback rather than the host's. The Makefile
uses `$(HOST_RUN)` for it, and `tests/test_gateway_exclusive.py` fails the build
if that ever reverts to `$(RUN)`.

It was routed through the container once, which broke `make gateway-start` with
a compose error and a message that said only "could not ask Docker". The subtler
half is why the redundancy did not save it: the container check *abstained*, but
the port check *answered* — about the wrong loopback. Two checks are only
independent if both can see the thing they are checking.

`desk doctor` runs in the container by default, so it cannot evaluate this guard
either. It now says so explicitly instead of reporting the Gateway as merely
unreachable, which had made a missing safety check look like an absent one.

## Known gaps

[`FOLLOWUPS.md`](FOLLOWUPS.md) lists what is deliberately missing and what
unblocks it — chiefly two free API keys (Finnhub, FRED), the regime tag that
depends on FRED, and the GDELT collection, which is the one gap whose cost
is irreversible.

## Corrections to the design docs, found while building Stage 0

The two design documents were written before any code existed, so a few claims
did not survive contact. Recording them here rather than silently diverging:

1. **Langfuse is not "one container."** Architecture §1 describes it that way,
   which was true of Langfuse v2. Self-hosted v4 is **six services** — web,
   worker, postgres, clickhouse, redis, minio — roughly 2 GB of images. It is
   still MIT and still free, but that is a different proposition on a laptop
   also running a 14B model. It sits behind a compose profile, and
   `docker/docker-compose.langfuse.yml` lists the fallbacks in preference order.
   The §1 reasoning survives: tracing was explicitly ranked a *bonus*, not the
   case for adopting LangGraph.

2. **Langfuse v4 runs in `events_only` mode**, which removes the legacy
   `/api/public/traces` and `/api/public/observations` REST endpoints. If you
   want to assert on spans programmatically, query ClickHouse's `events_full`
   table, not the REST API.

3. **The Gateway is no longer shared, and `trading-network` is gone.** Migration
   plan §0 originally had this repo join the ORB engine's Gateway over an
   external `trading-network`. That network was later removed from the machine,
   so `docker compose up execute` failed on missing infrastructure this repo
   does not own. Reversed: we run our own Gateway on our own
   **`trading-llm-network`**. §0 has been rewritten rather than patched.

4. **GDELT is rate-limited, and says so only in the 429 body.** The rule is one
   request per 5 seconds, with no `Retry-After` and no published quota page.
   Worse, a 429 has two causes wanting opposite responses — a momentary burst
   (retry) and a sustained per-IP block (stop and wait; an IP in that state was
   still refused after 150 s of backoff). The collector now paces on a clock
   gate, retries twice briefly, and then tells you to wait.

5. **GDELT's reachable window is still unverified.** §7.5 flags a contradiction
   in GDELT's own documentation and asks for it to be measured. `make
   gdelt-probe` measures it. That has *not* been run yet — GDELT was unreachable
   from the network this was built on, so the collector's live path is untested.
   Its parsing and merge logic is covered offline by `tests/test_gdelt_collect.py`.

## Back up `data/cache/gdelt`

It is gitignored, and it is the one thing here that cannot be regenerated.
GDELT's open API serves a rolling ~3-month window, so a day not collected is
gone permanently — and it is a day missing from the Stage 8 evaluation. Run
`make gdelt` daily (a cron entry once the output looks right), check `make
gdelt-status` for gaps, and make sure whatever backs up this machine includes
that directory.

## Rules that the tests enforce

These are not style preferences; each protects a decision that is otherwise easy
to erode one commit at a time.

`tests/test_layering.py` fails the build on the first four:

1. `graph/build.py` is the **only** module that imports `langgraph`.
2. `execution/` is the **only** package that imports `ib_async`.
3. Nothing outside `providers/` imports a provider module directly — everything
   goes through `providers/registry.py`.
4. No `langchain_core` chat models, no `ToolNode`, no `interrupt()`.

`tests/test_intent_blindness.py` fails it on the fifth, which is §8's and the
one a well-meaning refactor is most likely to "fix":

5. **Analysts are intent-blind.** No analyst sees the themes, the constraints,
   the risk limits or the drift table. Intent enters at the trader and no
   earlier, because the entire value of a bear researcher evaporates if every
   upstream report was primed with your thesis.

   Stage 6 made this load-bearing rather than theoretical: `prefetch` now puts
   the book and the drift table into `DecisionState`, so the data an analyst
   must not see sits in the object it reads. The guard checks both halves — no
   analyst module may *import* `render_drift` or anything from `intent/`, and
   the prompts each analyst actually sent are searched for the book's real
   numbers. It also checks the positive control, that the trader and fund
   manager *do* receive it, because a blindness test that passed by deleting
   the feature would be worthless.

`tests/intent/test_stage6_gate.py` fails it on the sixth:

6. **Sizing and compliance reach no broker and no network.** Asserted by
   replacing `socket.socket.connect` with a raising stub for the duration of
   the path, which is stronger than looking for `ib_async`: it proves the layer
   is not quietly reading a quote from anywhere.
