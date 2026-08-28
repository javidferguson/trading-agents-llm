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
| 1 | LLM router + structured output | not started |
| 2 | Providers, cache, metrics | not started |
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
```

Then check the environment is actually wired up:

```bash
make doctor
```

`make help` lists everything. The three checks worth knowing:

- `make check-ollama` — curls `/api/tags` **from inside the container**, which is
  where the `OLLAMA_HOST=127.0.0.1` trap actually bites (architecture §10).
- `make check-gateway` — one line telling you whether the IB Gateway owned by the
  ORB+GEX repo is up. This repo never starts one.
- `make toy-graph` — the Stage 0 exit gate: a two-node LangGraph run whose nodes
  both appear as spans in Langfuse.

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

3. **GDELT's reachable window is still unverified.** §7.5 flags a contradiction
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
