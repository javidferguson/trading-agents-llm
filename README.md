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
| 2 | Providers, cache, metrics | **2a+2b done** (bars, EDGAR, 60 metrics); 2c pending |
| 3 | Vertical slice → `proposal.json` | not started |
| 4 | Analysts + research debate | not started |
| 5 | Risk debate + fund manager | not started |
| 6 | Intent, sizing, compliance | not started |
| 7 | Execution + confirmation gate | not started |
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

It runs the Stage 0 gate in order — config, tests, environment, toy graph — and
stops at the first real failure. `make doctor` alone gives just the environment
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
- `make snapshot` — the Stage 2 exit gate: all 60 metrics for a symbol, each
  either populated or explicitly unavailable *with a reason*. No LLM.
  Needs `SEC_EDGAR_USER_AGENT` set, or EDGAR 403s without saying why.
- `make smoke` — the Stage 1 exit gate: a real `AnalystReport` out of a local 8B
  model, showing attempts, latency, tokens and the model digest.

The live model tests are deselected from `make test` (they add ~30s and need
Ollama up); `make test-smoke` runs them explicitly, and `make verify` includes
them.

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
to erode one commit at a time. `tests/test_layering.py` fails the build on all
four.

1. `graph/build.py` is the **only** module that imports `langgraph`.
2. `execution/` is the **only** package that imports `ib_async`.
3. Nothing outside `providers/` imports a provider module directly — everything
   goes through `providers/registry.py`.
4. No `langchain_core` chat models, no `ToolNode`, no `interrupt()`.
