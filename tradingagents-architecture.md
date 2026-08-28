# TradingAgents: architecture and design decisions

Design for a multi-agent LLM research desk, modeled on the TradingAgents paper
(arXiv 2412.20138), intended to live in **a separate repository** from this
ORB+GEX options engine.

This document exists so the reasoning survives the gap between projects. Most of
it is *why*, not *what* — the what is recoverable, the why is not.

**Status:** designed, not built. Design reviewed and revised — §1 reversed
(LangGraph is now in), §2 extended with a tool-promotion path, §7 rewritten
metrics-first, node 4 changed from social to positioning, and news-derived
sentiment added to §7.1 with X/Twitter and single-outlet scraping ruled out.

**Companion document:**
[`tradingagents-migration-plan.md`](tradingagents-migration-plan.md) — the
sequenced build plan, the carry list from this repo, and the per-stage exit
criteria. This document is *why*; that one is *in what order*.

---

## Scope

- **Equities and ETFs first.** Daily bars, BUY/SELL/HOLD on shares. The options
  strategy in this repo stays separate.
- **Local models by default**, hosted API for the one or two nodes where
  reasoning quality actually moves the outcome.
- **Every order requires human confirmation.** Paper account only.
- Driven by a stated **portfolio intent** rather than a watchlist.

---

## 0. The decision everything else follows from

**Split into two processes that communicate through a file, never a shared
event loop.**

```
[ decide ]  agents + LLM + free data  →  proposal.json  →  [ execute ]  human gate + ib_async
 no IB connection, no ib_async import                       no LLM, no network beyond IB
```

This solves five problems at once:

1. **Avoids the event-loop tangle.** `ib_async` installs its own loop policy and
   already needs `nest_asyncio`. Mixing that with a dozen concurrent LLM HTTP
   calls is a debugging tarpit. Separate processes, separate loops.
2. **The confirmation gate becomes a file you read**, not an `input()` racing a
   socket keepalive.
3. **Local model slowness stops mattering.** `decide` runs after the close;
   `execute` runs when you are at the keyboard.
4. **Replay is free.** `decide` in replay mode emits the same artifact type.
5. **IB can be down** while you research.

Everything below assumes this split.

---

## 1. Framework: LangGraph, as an orchestrator and nothing else

*Reversed from an earlier draft that argued for a plain `async def`. The
arguments against were mostly inherited from the LangChain era and did not
survive checking. What follows is the corrected reasoning, including the one
objection that did survive.*

**Use LangGraph for orchestration. Do not use it for anything else.**

### What was wrong with the case against

| Old objection | Status |
|---|---|
| "~40 transitive dependencies" | Wrong. `langgraph` 1.x pulls `langchain-core`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk`, `pydantic`, `xxhash`. Real weight, not 40 packages |
| "Explainability wants LangSmith to be pleasant" | Wrong. LangGraph emits OpenTelemetry. Langfuse (MIT, self-hosted, one container) gives the full trace tree with no SaaS account and no vendor lock |
| "Per-node model routing is a dict either way" | True, and therefore not an argument for either side |
| "Its value-add is `ToolNode` and ReAct loops, which we must not use" | Half true. That is its *headline* value-add, not its only one — see below |

### What it actually buys, honestly ranked

1. **Durable node-level resume.** This is the real prize and the earlier draft
   missed it entirely. With Ollama at ~30 s/call, a run is 3–8 minutes. Failing
   at node 11 of 14 and resuming beats re-running from node 0, and during
   prompt development you will do this many times a day.
2. **Node-level trace spans**, free, into Langfuse or any OTel collector. Note
   this is *available without LangGraph* via plain decorators — so tracing alone
   would not justify the dependency. It is a bonus on top of (1), not the case.
3. Fan-out, conditional edges, streaming to a future UI. All marginal against
   `asyncio.gather` and a `for` loop. Do not let these be the reason.

### The objection that survived, and the rule it produces

**LangGraph's checkpointer is not the replay you want.** Checkpointing resumes
*a thread* mid-graph. `mode="replay"` means "re-run 2024-03-15 against frozen
data with a different model"; `mode="replay_llm"` means "re-run against recorded
LLM responses." Neither is a checkpointer feature and neither ever will be.

> **Rule:** `DecisionState` JSONL is the source of truth for replay and audit.
> The LangGraph checkpointer exists **only** for crash-resume within a single
> live run. Two mechanisms, two jobs. Do not merge them, and do not let a future
> refactor "simplify" by deleting the JSONL.

### The three constraints

1. **Orchestration only.** Own LLM router, own Pydantic schemas, own
   `structured()` with the repair turn (§6). Do **not** route calls through
   `langchain-core` chat models — that is where the abstraction tax actually
   lives, and it makes Ollama's `format=<json_schema>` awkward for no gain. A
   LangGraph node body should call *our* client.
2. **Pin exact versions.** Not theoretical: `langgraph-prebuilt` 1.0.2 shipped a
   breaking change under loose constraints and took dependents down with it. Pin
   `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langchain-core`
   to exact versions and upgrade deliberately.
3. **LangGraph never enters the `execute` process.** §0's split is unrelated to
   this decision and is not up for revision. In particular: **no `interrupt()`
   for the human gate.** The file-based gate is deliberately better, and
   LangGraph's human-in-the-loop primitives will tempt someone to collapse the
   two processes into one. That would reintroduce the event-loop tangle §0
   exists to avoid.

### The signature, still mandatory

```python
async def node(state: DecisionState, ctx: NodeContext) -> dict[str, Any]:  # partial-state patch
```

Originally the hedge for adopting LangGraph later; now it is simply what
LangGraph wants. It stays valuable in the other direction — a node that only
reads `state` and returns a patch can be unit-tested, replayed, and if necessary
un-adopted without a rewrite.

**`DecisionState` is a Pydantic model, used directly as the graph state.**
LangGraph supports this. Fields that accumulate (`llm_calls`, `errors`,
`violations`, debate `turns`) need reducer annotations; everything else is
last-write-wins.

---

## 2. The rule that makes weak local models usable

**No tool-calling, ever.**

The paper's "~20 tool calls" are not agentic decisions — they are
`get_price_history`, `get_news`, `get_financials`. There is no judgment in
choosing them.

So: **deterministic prefetch, then constrained summarization.**

1. `build_market_snapshot(symbol, as_of)` fetches everything from Python, with
   retries and caching. No LLM involved.
2. Every LLM node receives rendered pre-computed facts and returns **JSON
   constrained by a Pydantic schema** — Ollama's `format=<json_schema>`,
   Anthropic's tool-schema forcing.
3. **All arithmetic happens in `indicators.py`.** RSI, ATR, drawdown, YoY
   growth, position sizing — never in a prompt.

An 8B model will confidently produce a wrong RSI. It is genuinely good at "RSI
is 71 and price is at the upper Bollinger band — that's stretched." This single
rule is the difference between local models being usable and useless here.

### Keeping the door open to tool-calling

"No tool-calling" is a statement about *today's* models, not a permanent
architectural commitment. When a local model can be trusted to select fetches —
or when a run routes to a hosted model that already can — the prefetch functions
should be promotable to tools without touching their call sites.

Three cheap things now make that a config change later:

1. **One registry.** Every fetch lives in `providers/registry.py` and is reached
   through it. Nothing imports a provider module directly.
2. **Tool-shaped signatures.** Fully typed parameters, no `**kwargs`, no
   positional-only tricks, JSON-serialisable returns, and a docstring written as
   if it were already the tool description a model would read. Then the schema
   is *generated* from the signature rather than hand-written twice.
3. **`as_of` on every one of them**, always — which §7 requires anyway. A tool a
   model can call at an arbitrary moment is exactly where look-ahead bias would
   creep back in, so the point-in-time guard has to live in the function, not in
   the caller.

The upgrade is then: generate schemas from the registry, hand them to a
tool-capable node, keep the deterministic path as the fallback and as the
replay substrate. Nothing above needs rewriting.

---

## 3. Nodes

▪ = no LLM, ● = LLM

```
▪ 0  prefetch            build_market_snapshot()
▪ 1  memory_recall       past reflections for (symbol, regime)
● 2  market_analyst      ─┐
● 3  news_analyst         ├─ LangGraph parallel fan-out, 4-way
● 4  positioning_analyst  │
● 5  fundamentals_analyst─┘
   ── research debate, rounds 1..N ──
● 6  bull_researcher
● 7  bear_researcher
● 8  research_facilitator   (also emits should_continue)
● 9  trader                 -> TraderProposal (BUY/SELL/HOLD + conviction)
   ── risk debate, rounds 1..M ──
● 10 risky_debater
● 11 neutral_debater
● 12 safe_debater
● 13 fund_manager           -> FinalDecision (approve / adjust / veto)
▪ 14 compliance             deterministic veto vs PortfolioIntent
▪ 15 persist                write DecisionState + proposal.json
```

14 LLM calls at N=M=1. Matches the paper's ~11–14.

**Node 4 was `social_analyst` in the earlier draft.** It is now
`positioning_analyst`, reading short interest, short-sale volume and insider
transactions — all free, all point-in-time, all far better behaved than retail
social sentiment (§7). `social_analyst` survives as a swappable alternative
occupant of the same slot, shipping disabled. The call count is unchanged, so
nothing downstream moves.

**Node 3 `news_analyst` also carries sentiment**, rather than sentiment getting
a node of its own. It receives headlines *and* the pre-computed news-tone and
coverage-volume percentiles (§7.1). A separate sentiment node would have read
the same articles as the news node and then agreed with it — which is precisely
the "three agents agreeing is not three pieces of evidence" failure in §15.2.
Call count still 14.

---

## 4. State schema

`models/state.py`, Pydantic v2. Abbreviated:

```python
class Evidence(BaseModel):
    source: str; url: str | None; as_of: datetime; excerpt: str  # <=300 chars

class AnalystReport(BaseModel):
    kind: Literal["market","news","positioning","fundamentals","social"]
    stance: Literal["bullish","bearish","neutral"]
    confidence: float = Field(ge=0, le=1)
    summary: str                      # <=200 words
    key_points: list[str]             # 3-6
    evidence: list[Evidence]
    data_gaps: list[str]              # what it could NOT see -- feeds calibration
    parse_failed: bool = False

class DebateTurn(BaseModel):
    round: int; speaker: str; claim: str
    supporting_report_kinds: list[str]
    rebuts: str | None
    new_information: bool             # facilitator-verified; drives the novelty stop

class DebateTranscript(BaseModel):
    turns: list[DebateTurn] = []
    rounds_completed: int = 0
    stop_reason: Literal["max_rounds","converged","no_novelty","budget","error"] | None = None

class ResearchVerdict(BaseModel):
    winner: Literal["bull","bear","tie"]
    thesis: str
    strongest_counterargument: str    # dissent is REQUIRED, never dropped
    confidence: float

class TraderProposal(BaseModel):
    action: Literal["BUY","SELL","HOLD"]
    conviction: float = Field(ge=0, le=1)
    target_weight_pct: float          # desired % of equity, not share count
    horizon_days: int
    rationale: str
    invalidation: str                 # "what would prove this wrong"
    intent_alignment: str             # which theme/gap this closes

class FinalDecision(BaseModel):
    schema_version: int = 1
    action: Literal["BUY","SELL","HOLD"]
    symbol: str
    conviction: float
    target_weight_pct: float
    order_type: Literal["LMT","MKT"] = "LMT"
    limit_offset_bps: int = 10
    stop_loss_pct: float | None
    take_profit_pct: float | None
    horizon_days: int
    rationale: str
    dissent: str                      # preserved bear/safe case
    invalidation: str
    expires_at: datetime              # a stale proposal cannot be executed
    fund_manager_adjustment: str | None

class DecisionState(BaseModel):
    schema_version: int = 1
    run_id: str; symbol: str
    as_of: date                       # decision date -- the replay anchor
    created_at: datetime
    mode: Literal["live","replay","replay_llm"]
    config_hash: str; prompt_pack_version: str
    intent: PortfolioIntent
    portfolio: PortfolioSnapshot
    snapshot: MarketSnapshot
    regime: RegimeTag                 # see §7 -- the memory-retrieval key
    memory_hits: list[MemoryHit] = []
    analyst_reports: dict[str, AnalystReport] = {}
    research_debate: DebateTranscript = DebateTranscript()
    research_verdict: ResearchVerdict | None = None
    trader_proposal: TraderProposal | None = None
    risk_debate: DebateTranscript = DebateTranscript()
    final_decision: FinalDecision | None = None
    violations: list[Violation] = []
    llm_calls: list[LLMCallRecord] = []   # prompt hash, model+digest, RAW response, tokens, ms, $
    errors: list[NodeError] = []
```

Three details that pay for themselves:

- **`LLMCallRecord` stores the raw response**, enabling `mode="replay_llm"`:
  re-run the pipeline against recorded outputs to test downstream changes at
  zero cost and zero latency. Use this constantly during development.
- **`dissent` and `invalidation` are required fields** on `FinalDecision`. This
  is where the paper's explainability claim actually cashes out — both surface
  in the confirmation prompt.
- **The accumulating fields need LangGraph reducers.** `llm_calls`, `errors`,
  `violations`, and each `DebateTranscript.turns` are appended to by nodes that
  run in parallel; annotate them with an add-reducer or the four analysts will
  silently overwrite each other's `llm_calls`. Everything else is
  last-write-wins. This is the one place LangGraph's state model needs care, and
  it is worth a test that runs the fan-out and asserts four records land.

---

## 5. Debate termination

Four conditions. **Budget always wins.**

1. `rounds_completed >= cfg.max_rounds` (default **1**, not 2 — start cheap)
2. Facilitator returns `converged=True` with a reason
3. **Novelty guard**: every turn this round has `new_information=False`, or a
   speaker's claim has >0.85 Jaccard overlap with its previous round. Local
   models loop and restate; this is not optional.
4. `budget.exhausted()`

Conditions 1 and 4 are hard caps enforced in Python. Never let the model alone
decide when to stop.

**Failure direction is always HOLD.** Any node error, budget exhaustion, or
schema-parse failure after one repair turn →
`FinalDecision(action="HOLD", rationale="degraded: <reason>")`. Never fail
toward a trade.

---

## 6. Model routing

`config/models.yaml`:

```yaml
providers:
  ollama:    {base_url: "${OLLAMA_BASE_URL}", timeout_s: 300}
  anthropic: {api_key_env: ANTHROPIC_API_KEY}
profiles:
  quick:       {provider: ollama,    model: "qwen3:8b",   temperature: 0.3, num_ctx: 8192}
  deep_local:  {provider: ollama,    model: "qwen3:14b",  temperature: 0.5, num_ctx: 16384}
  deep_hosted: {provider: anthropic, model: "<model-id>", temperature: 0.5, max_tokens: 2000}
nodes:
  market_analyst: quick
  news_analyst: quick
  positioning_analyst: quick
  fundamentals_analyst: quick
  bull_researcher: quick
  bear_researcher: quick
  research_facilitator: deep_local
  trader: deep_local
  risky_debater: quick
  neutral_debater: quick
  safe_debater: quick
  fund_manager: deep_hosted      # the one node worth paying for
  reflection: deep_local
presets:
  all_local:  {"*": quick}
  all_hosted: {"*": deep_hosted}   # for eval / strategy development
```

`structured()` handles: schema-constrained decode → validate → on failure, one
repair turn with the validation error appended → on second failure, return
`schema.degraded()` with `parse_failed=True`. A degraded analyst report beats a
crashed 8-minute run.

---

## 7. Metrics, and the free sources reverse-engineered from them

No paid market-data subscription assumed. `indicators.py` computes everything
numeric; providers only fetch.

**Decide the metrics first, then find sources that can serve them.** The earlier
draft did this backwards — it picked providers and took whatever indicators fell
out, which is how you end up with six flavours of moving average and no measure
of profitability. What follows is the metric set the analysts actually need,
then the sources chosen to serve it.

**Scope: per-symbol, plus a shared macro regime tag.** Every metric below is
absolute or measured against the symbol's own history (percentile over 1–5
years). There is deliberately **no peer-universe cross-section** — no "cheap
relative to sector" z-scores — because that requires a nightly universe-wide
fetch and a local store, which is a project of its own. The two exceptions are
relative strength vs SPY and vs the sector ETF, which need one extra series
each, not a universe. Note the limitation honestly in prompts: the fundamentals
analyst is seeing absolute levels and the symbol's own trend, not a peer
comparison.

### 7.1 The metric set

▲ = computed by us in `indicators.py`, never by a model (§2).

**Price and trend** — ▲ from daily OHLCV

| Metric | Why it earns a slot |
|---|---|
| SMA 20/50/200, price vs each, 50/200 cross state | Trend regime, and the B3 baseline is built from it |
| **12-1 momentum** (12-month return excluding the most recent month) | The academically standard momentum definition. The excluded month matters — recent-month reversal contaminates it |
| 1-month reversal | Opposes momentum at short horizon; keeping both stops the model conflating them |
| 52-week percentile, max drawdown 1y | Where in its own range it sits |
| **Relative strength vs SPY and vs sector ETF** | The largest gap in the earlier draft. Absolute return tells you almost nothing in a trending market |

**Risk and liquidity** — ▲

| Metric | Why |
|---|---|
| ATR-14 | Feeds stop distance, so it feeds position sizing (§9) |
| Realized vol 20d / 60d, ratio between them | Vol expanding or contracting |
| Beta vs SPY, downside deviation | Residualising returns in eval (§11) needs beta anyway |
| 20d dollar ADV, volume z-score, Amihud illiquidity | ADV participation is a hard compliance check (§9) |

**Mean reversion** — ▲

RSI-14, MACD, Bollinger %B, distance from 20d VWAP, EMA 12/26.

**Value** — ▲ from SEC EDGAR facts

Earnings yield (E/P), FCF yield, EV/EBIT, book/price, sales/price. Yields rather
than multiples: they stay finite through negative earnings, which multiples do
not, and a small model handles "-2% earnings yield" far better than "P/E of
-48".

**Quality** — ▲ from SEC EDGAR facts

| Metric | Why |
|---|---|
| **Gross profitability (gross profit / assets)** | Novy-Marx. About as well-evidenced as value, and nearly free to compute |
| **Accruals** (net income − operating cash flow, scaled by assets) | The best-documented *avoid* signal in the free-data universe. Cheap, and it catches the thing news sentiment never will |
| ROIC, ROE, net debt/EBITDA, interest coverage, current ratio | Standard solvency and returns |
| **Piotroski F-score** | All nine binary components are computable from EDGAR alone. A single 0–9 integer is exactly the kind of input a small model uses well |

**Growth and dilution** — ▲ from SEC EDGAR facts

Revenue YoY and 3y CAGR, EPS YoY, gross and operating margin trend, and
**share-count change** — buyback vs dilution is free, point-in-time, and
routinely ignored.

**Events and estimates** — fetched, partially point-in-time

Earnings calendar and blackout window (also a hard compliance input, §9), EPS
surprise history, analyst recommendation trend.

> **Caveat to write into the prompt, not just the code:** post-earnings-
> announcement drift has measurably weakened in recent US data. Measure earnings
> surprise; do not build a node around it or let a prompt assert it as a law.

**Positioning** — fetched, point-in-time

Short interest and days-to-cover (semi-monthly), daily short-sale volume ratio,
insider buy/sell counts and net dollar value from Form 4. **This is what node 4
reads instead of Reddit.**

**Sentiment and attention** — fetched, then ▲ normalised

The premise is sound: how people feel about an asset does affect how it trades.
The difficulty is entirely in sourcing it without buying a feed, and in not
getting the *sign* wrong. Four measurable constructs, deliberately kept apart
because they behave differently:

| Construct | Metric | Source |
|---|---|---|
| **News tone** | Mean article tone for the company over 1d / 7d / 30d, and its percentile vs the symbol's own trailing year | **GDELT** `TimelineTone` |
| **Coverage volume** | Article count per day, z-scored vs the symbol's own 90-day baseline | **GDELT** `TimelineVol`, cross-checked against Finnhub article counts |
| **Search attention** | Google Trends interest for the ticker and company name, percentile vs own history | pytrends — *optional, see caveats* |
| **Market-wide sentiment** | CBOE equity put/call ratio and its percentile; VIX percentile | **CBOE** daily statistics; FRED |

Three rules that matter more than the metrics themselves:

1. **Feed percentiles, never raw tone, and never an adjective.** A tone score of
   `-1.7` means nothing to an 8B model; "in the 12th percentile of this symbol's
   own last year" means something. And see the sign warning below before writing
   any prompt that describes sentiment in words.

2. **The sign is not the obvious one, and a small model will get it wrong.**
   Extreme bullish sentiment and abnormally high attention are, at short
   horizons, better documented as *contrarian* signals than confirming ones —
   attention-induced buying tends to precede reversal. Tell an 8B model
   "sentiment is very positive" and it will say BUY, confidently, every time.
   **So do not editorialise in the prompt.** Give the percentile and the
   direction of change; let the bull and bear researchers argue about what it
   means. This is one of the clearest cases where §2's "compute in Python, judge
   in the model" split has to be enforced at the *prompt* level too.

3. **Tone is largely priced in for large caps.** Expect a weak signal, and
   expect the calibration plot (§11) to show it. It earns its place as a
   *tiebreaker and a risk flag* — an unusual spike in coverage volume is a
   better reason to pay attention than the tone of that coverage — not as a
   primary driver.

**Macro** — fetched once per run, shared across symbols

VIXCLS, DGS10, DGS2, T10Y2Y, DFF, CPIAUCSL, UNRATE, plus two the earlier draft
missed: **BAMLH0A0HYM2** (high-yield OAS — the cleanest free risk-on/risk-off
read) and **NFCI** (Chicago Fed financial conditions).

### 7.2 The regime tag

Memory retrieval (§3 node 1) keys on `(symbol, regime)`, so `regime` needs a
definition tight enough to match on. Three axes, twelve buckets:

```
vix_bucket   : VIXCLS  <15 | 15-25 | >25
curve        : T10Y2Y  >= 0 | < 0
trend        : SPY above | below its 200-day
```

Coarse on purpose. Finer buckets mean fewer historical matches, and with 60–90
shadow sessions (§11) you have very little to match against. Store the raw
values alongside the tag so the buckets can be re-cut later without re-fetching.

**Market-wide sentiment does not get a fourth axis.** Adding put/call would
double the bucket count to 24 and halve an already thin match rate — the
opposite of what memory retrieval needs. Carry the put/call ratio and its
percentile as **raw fields alongside** the tag, available to prompts and to the
eval harness but not part of the retrieval key. Revisit only if the shadow-mode
corpus ever gets large enough that 12 buckets are too coarse, which will take
considerably longer than 90 sessions.

### 7.3 Providers, chosen to serve the above

| Need | Primary | Fallback | Key | Notes |
|---|---|---|---|---|
| Daily OHLCV | **Stooq** — no key, decades of daily history | **IB `reqHistoricalData`** | none | **yfinance is demoted to third.** It is an unofficial scrape that has been structurally unreliable since Yahoo's 2025 redesign — rate limits, IP blocks, schema churn. IB is the authoritative fallback and the Gateway is already running |
| Fundamentals | **SEC EDGAR** XBRL `companyfacts` | Finnhub `/stock/metric` | none for EDGAR | **The best free source in the stack.** Every fact carries its `filed` date, so it is genuinely point-in-time rather than approximately so. Requires a `User-Agent: Name email` header — a missing one is the usual cause of a 403 — and 10 req/s |
| News | **Finnhub** `/company-news` | SEC 8-K feed; Yahoo RSS | free, 60/min | Finnhub is date-rangeable (1y history on free), which is exactly what replay needs. **RSS is not date-rangeable, so it is `supports_point_in_time=False` and must raise in replay** |
| Events/estimates | **Finnhub** earnings calendar, surprises, recommendation trends | — | free, 60/min | |
| Positioning | **FINRA** short interest + daily short-sale volume; **SEC EDGAR** Form 4 | — | none | Short interest carries a settlement date; Form 4 carries a filing date. Both point-in-time |
| **News tone / volume** | **GDELT** DOC 2.0 (`TimelineTone`, `TimelineVol`) | Finnhub article counts | **none** | Open API, no key, no auth. Aggregates thousands of outlets across 100+ countries — see §7.6 on why breadth is the whole point. **Rolling 3-month window only**, which drives a build-order requirement (§7.5) |
| Market sentiment | **CBOE** daily options statistics — equity put/call ratio | — | none | Free daily files with an archive back to 2006. The cleanest free sentiment series in the stack |
| Search attention | pytrends (Google Trends) | — | none | **Optional.** Unofficial, rate-limited, and its values are *relative to the requested window*, so the same date returns different numbers depending on the query range — a genuine replay hazard. If used at all: snapshot at `as_of`, cache, and never re-query a cached date |
| Macro | **FRED** | — | free key | Generous limits, one fetch per run |

Deliberately **not** used:

- **Alpha Vantage** — 25 requests/day now, unusable for anything but a one-off
  tiebreaker.
- **X / Twitter.** As of February 2026 there is **no free tier for new
  developers**; access is pay-per-use at ~$0.005 per post read, and
  **full-archive search is Enterprise-only, starting around $42,000/month**.
  The archive is precisely what replay would need, so X fails on cost *and* on
  point-in-time grounds simultaneously. Trending topics are also regional and
  rarely ticker-specific. Ruled out, not deferred.
- **Scraping individual news sites** (MSNBC, CNBC, and similar). Four problems
  at once: a single outlet is a house-view sample rather than a measurement;
  general-interest outlets carry almost no per-ticker coverage; HTML schemas
  churn, which is the exact failure mode that demoted yfinance; and there is no
  historical archive, so replay is impossible. GDELT already does this job
  properly, across thousands of outlets, with a stable API and no terms-of-service
  question.
- **NAAIM Exposure Index** — moved to subscription access on 1 August 2026.
- **AAII Sentiment Survey** — the current weekly reading is public, but the
  historical spreadsheet is behind membership, so it cannot anchor a percentile.
  Usable as a live-only curiosity; not worth a provider.
- **Finnhub `/news-sentiment`** — its scored sentiment sits under alternative
  data, which is a Premium entitlement. Finnhub's free `/company-news` gives
  headlines and article counts, which is what we actually use it for. Verify
  before assuming otherwise.

### 7.4 Point-in-time discipline

Every provider method takes `as_of`. Cache key =
`sha256(provider, method, sorted(kwargs), as_of)`. **In `mode="replay"` the
cache is the only permitted source** — HTTP is hard-disabled and any provider
with `supports_point_in_time=False` raises. Look-ahead bias is the number one
way agentic backtests produce fictional Sharpe ratios.

Record which source served each field. When Stooq fails over to IB the numbers
differ slightly (adjustments, session boundaries) and you will want to know
which one a decision was made on.

### 7.5 What free data cannot give you

State these as known biases rather than discovering them at Stage 8:

1. **No survivorship-bias-free universe and no delisting returns.** Every free
   source lists companies that exist today. A backtest over any universe you
   assemble now silently excludes everything that went to zero, which inflates
   returns. This is the single largest threat to the §11 numbers.
2. **No point-in-time index membership.** "Was this in the S&P 500 in 2023?" is
   not answerable free.
3. **No point-in-time consensus estimates.** Current estimates are available;
   what consensus *was* on a past date is paywalled everywhere. So a
   surprise-vs-consensus metric is honest live and dishonest in replay — compute
   it live only, and have the provider raise in replay.
4. **No historical options greeks or open interest** — carried over from the ORB
   engine (§15.10) and unchanged.
5. **No retroactive news sentiment.** GDELT's open API serves a **rolling
   three-month window**. You cannot backfill tone for a replay of 2024, and the
   full GDELT GKG archive that would let you is a bulk-download project of its
   own, not an API call.

   *Caveat on that claim:* GDELT's own announcement says the API searches "a
   rolling window of the last 3 months of coverage" while also referring to an
   index reaching back to 1 January 2017, and those two statements are not
   obviously reconcilable. **Verify the actual reachable `startdatetime` at
   Stage 2** rather than trusting this document. The recommendation below holds
   either way: if the window really is three months, caching from day one is
   mandatory; if more turns out to be reachable, caching from day one cost
   nothing.

   > **Build-order consequence, and it is the reason this is listed here rather
   > than in §13:** start the GDELT cache accumulating **on day one of Stage 2**,
   > long before any node reads it. Every day not cached is a day permanently
   > missing from the Stage 8 evaluation. This costs one cron entry now and is
   > unrecoverable later.

   Any replay earlier than the cache start must return sentiment fields as
   `None` **with a stated reason** — never zero, and never a neutral default. A
   silent zero is indistinguishable from genuinely neutral coverage, and it will
   quietly corrupt the ablation that decides whether sentiment earns its place.
6. **No free historical X/Twitter.** Full-archive search is Enterprise-priced.
   Live sampling would be affordable but produces a signal that cannot be
   replayed, which makes it unevaluable.

### 7.6 Three kinds of "sentiment", and why only two are used

The earlier draft treated sentiment as one thing, decided it was untrustworthy,
and shipped a Reddit/PRAW analyst turned off by default. That collapsed a
distinction worth keeping. There are three separable constructs here, and they
differ enormously in how measurable and how manipulable they are:

| | What it measures | Free? | Point-in-time? | Manipulable? | Verdict |
|---|---|---|---|---|---|
| **Revealed positioning** | What people did with money — short interest, short-sale volume, Form 4 | Yes | Yes, dated filings | Barely — it is regulated disclosure | **In**, node 4 |
| **Measured coverage** | How much the world is talking, and in what tone — GDELT across thousands of outlets | Yes | Within a 3-month window | Hard at aggregate scale | **In**, as metrics into node 3 |
| **Self-reported chatter** | What retail says on Reddit / X | Reddit yes, X no | Poorly | **Trivially, and by design** | **Out** |

The dividing line is **breadth**. A subreddit or a trending hashtag is a small,
self-selected, actively promoted sample — the noisiest input available and the
one most likely to make a small model enthusiastic about exactly the wrong
thing. An aggregate tone score across thousands of outlets in 100+ countries is
a *measurement*, and moving it requires moving the news cycle rather than
organising a Discord.

So the instinct to reach for news rather than social media is the right one —
it just needs an aggregator rather than any individual outlet, for the reasons
in §7.3.

`social_analyst` stays in the tree as a swappable alternative occupant of slot
4, shipping off, and turns on only if an ablation shows lift. X is not an option
for it at any setting (§7.3).

---

## 8. Portfolio intent

`config/portfolio-intent.yaml` — objective (horizon, benchmark, prose), `risk:`
(per_trade_risk_pct, default_stop_pct, max_position_pct, max_sector_pct,
min_cash_pct, max_gross_exposure_pct, max_positions, min_trade_usd,
max_order_shares), `universe:` (include/exclude/allow_shorts), `themes:` (name,
target_weight_pct, conviction, thesis, exemplars), optional `targets:` with
bands, free-text `constraints:`, and `cadence:` (max_decisions_per_day,
proposal_ttl_hours).

**Consumed through four distinct channels, deliberately not one:**

1. **Deterministic pre-filter (no LLM).** Universe minus exclusions minus
   earnings blackout minus at-max-positions → today's candidates, computed
   before any model runs. Saves money and removes a class of hallucination.
2. **Drift table (no LLM).** `intent_engine.compute_gaps()` → per symbol and
   theme: current weight, target, band, gap in % and USD. The Trader node gets
   this as a table. **Its job is choosing which gap to close, not inventing
   allocations.** This is the biggest reliability win — it converts an
   open-ended "how much should we buy" into a bounded "close this gap or don't",
   which small models handle far better.
3. **Prompt context (LLM).** Objective description, theme theses and free-text
   constraints go into the **Trader and Fund Manager system prompts only**.
4. **Hard veto (no LLM).** Everything under `risk:` is enforced by
   `compliance.py` *after* sizing.

**Analysts are intentionally intent-blind.** Do not tell the fundamentals
analyst "we are bullish on AI infrastructure" before it reads the 10-K. The
entire value of a bear researcher evaporates if every upstream report was primed
with your thesis. Intent enters at the Trader node and no earlier. This deserves
a code comment so nobody "helpfully" fixes it later.

---

## 9. Decision → broker

Three layers, so the first two run with no broker at all:

```
FinalDecision (broker-agnostic) → OrderPlan (sizing + compliance) → ib_async Contract/Order
```

**Sizing** — deterministic, minimum of four caps:

```
conviction_w    = target_weight_pct * conviction
cap_position    = intent.risk.max_position_pct
cap_risk        = intent.risk.per_trade_risk_pct / stop_dist
target_w        = min(conviction_w, cap_position, cap_risk)
delta_notional  = equity * target_w/100 - pf.position_value(symbol)
```

Then reject dust (`< min_trade_usd`), clamp to `max_order_shares` and 90% of
buying power.

**Compliance** — post-trade-state checks against max_position_pct,
max_sector_pct, min_cash_pct, max_gross_exposure, max_positions, universe
membership, earnings blackout, max_decisions_per_day, ADV participation.
**Any violation blocks the order regardless of what the fund manager decided.**
Two-key system: LLM proposes, Python vetoes. This is where trust comes from.

**Execution** — three rules carried from the options engine:

- **`assert_paper_account()` after connect *and* again immediately before every
  `placeOrder`.** The port is not a safety guarantee; a socat or env-var mistake
  can point a "paper" port at live. `managedAccounts()` is the real gate.
- **`ib.whatIfOrder()` before the human sees the prompt** — free, places
  nothing, returns margin impact and commission. Put those numbers *in* the
  confirmation.
- **Marketable `LimitOrder`, never `MarketOrder`.** With delayed data you are
  looking at a 15-minute-old price.

**Confirmation** requires typing the **ticker symbol**, not `y`. A yes/no prompt
is answered by muscle memory. Show the dissent, the invalidation condition, and
the whatIf numbers. There is deliberately no config flag to disable the gate.

---

## 10. Ollama on macOS: run it natively

**Do not containerize Ollama on macOS.** Docker Desktop runs a Linux VM with no
Metal passthrough, so a containerized Ollama is CPU-only inside a VM with a
fraction of your RAM — roughly **5–15× slower** than native with Metal. A
14-call pipeline that takes 4 minutes natively takes 40+ containerized. That is
the difference between a tool you use and one you abandon.

The app container reaches host Ollama via
`OLLAMA_BASE_URL=http://host.docker.internal:11434` plus
`extra_hosts: ["host.docker.internal:host-gateway"]` for Linux portability.

**The gotcha that costs an hour:** Ollama binds `127.0.0.1:11434` by default,
which a container cannot reach. Set `OLLAMA_HOST=0.0.0.0:11434` on the host
daemon. Add a check target that curls `/api/tags` **from inside the container**
so this is diagnosed in seconds rather than via stack traces.

Still ship a containerized Ollama behind a **compose profile** so the same file
works on a Linux/NVIDIA box and in CI. Set `OLLAMA_KEEP_ALIVE=30m` (otherwise
the model unloads between nodes and you pay a multi-second reload per call) and
`OLLAMA_NUM_PARALLEL=4` (otherwise the four-analyst fan-out silently
serializes).

---

## 11. Evaluation: the go/no-go

The purpose is **not** to prove alpha — on three months you cannot. It is to
detect disqualifying badness, check calibration, and answer one question:
**does the multi-agent structure beat a single LLM call?**

| # | Baseline | Why |
|---|---|---|
| B0 | Buy-and-hold SPY | The bar you must clear to justify existing |
| B1 | Always HOLD | Isolates whether trading at all helps |
| B2 | Random, turnover-matched, **200 seeds** | Gives a null *distribution*, not a point. Without this you cannot tell skill from noise |
| B3 | Deterministic rule (50/200 SMA + RSI) | The cheapest thing that could work |
| **B4** | **Single LLM call**, same snapshot, one prompt | **The one that matters** |
| A | Full 14-node pipeline | |

**If A does not beat B4 by a margin outside B2's noise band, the multi-agent
layer is not earning its 14× cost — ship B4.** Build the harness so this
comparison is a one-line config change, and be genuinely willing to act on the
answer. Most multi-agent systems fail this test.

Scoring: forward returns at 1/5/20 days; hit rate; **information coefficient**
(Spearman of stated conviction vs forward return — if conviction does not
correlate with outcomes, the model's confidence is decorative); turnover; equity
curve using the real sizing and compliance code, not idealized fills. **Report
residual return vs SPY**, not raw, or you will conclude the agents are geniuses
in an up market. Block bootstrap (5-day blocks) for confidence intervals. A
**calibration plot** (realized hit rate bucketed by stated confidence) is cheap
and usually the most damning diagnostic.

**Shadow mode is the primary tool**: run `decide` daily on live data with
`execute` never called, journal everything, accumulate 60–90 sessions.
Out-of-sample and leakage-free — worth more than any replay.

Ablations, one flag each: `positioning.enabled=false`, `sentiment.enabled=false`,
`research_rounds=0`, `risk_rounds=0`, `memory.enabled=false`,
`trader→deep_hosted`, `all_local` vs `all_hosted`. Add `social.enabled=true` as
the ablation that decides whether the disabled Reddit node (§7.6) ever earns its
place.

**The sentiment ablation deserves a specific prediction, made before it runs:**
news tone is largely priced in for liquid large caps, so expect little or no
lift from tone, and *more* from abnormal coverage volume than from its polarity.
Writing the prediction down first is what makes the result informative rather
than a rationalisation — and the calibration plot is where it will show up.

**Report the survivorship-bias caveat (§7.5) alongside every number in this
section.** It is not a footnote; it is the reason a plausible-looking equity
curve may be fiction.

---

## 12. Cost and latency

| Config | Per symbol per decision |
|---|---|
| Native Ollama, 8B q4, parallel=4 | **~3–8 min**, $0 |
| Containerized Ollama on macOS | 20–60+ min — not viable |
| Hybrid (fund_manager hosted only) | ~3–8 min + **~$0.03** (~$40/yr at 5 symbols daily) |
| All 14 calls hosted | seconds + **~$0.40** (~$500/yr daily; a 90-day × 5-symbol replay ≈ $180) |

*Order-of-magnitude. Re-derive against current per-Mtok rates. Prompt caching on
the repeated system prompts cuts input cost materially — wire it on day one.*

Mitigations by leverage: **run after the close** (latency stops mattering, and
it is free); cache `AnalystReport` per `(symbol, as_of, prompt_version,
model_digest)`; use `mode="replay_llm"` for downstream development; enforce
`RunBudget(max_llm_calls=20, max_wall_s=900, max_usd=0.50)`; default both debate
rounds to 1; run symbols sequentially overnight (parallel symbols thrash
Ollama's model cache).

---

## 13. Build order

| Stage | Deliverable | Gate |
|---|---|---|
| 0 | pyproject/uv, `src/` layout, pinned LangGraph, Langfuse container | imports clean; a two-node toy graph traces end to end |
| 1 | `llm/` router + Ollama structured output + `models.yaml` | smoke test returns a valid Pydantic object |
| 2 | providers + cache + limiter + `indicators.py` + `snapshot.py` | `snapshot SYMBOL=SPY` prints a full `MarketSnapshot` with every §7.1 metric populated or explicitly `None`, no LLM |
| **3** | **Vertical slice: market_analyst → trader → `FinalDecision` → proposal.json** | **A sane, schema-valid decision end to end. Highest-value de-risking step** |
| 4 | 3 remaining analysts + bull/bear/facilitator + termination | Debate stops for the right reason every time; fan-out reducer test passes (§4) |
| 5 | Risk trio + fund_manager + `deep_hosted` routing | Full 14-node run under budget |
| 6 | `PortfolioIntent` + intent_engine + sizing + compliance + journal | `OrderPlan` produced, zero IB contact |
| 7 | `execution/` — safety, whatIf, limit orders, confirmation gate | One paper trade, human-confirmed, `DU`-asserted |
| 8 | `eval/` — replay, B0–B4, scoring, calibration | **The B4 comparison. Go/no-go** |
| 9 | `memory.py` + `reflect.py` | Reflections retrieved and measurably change decisions |

Stage 3 first, always. If market-analyst → trader → proposal does not produce
something sane, more agents will not fix it.

The detailed sequencing, the file-by-file carry list from the ORB engine, and
the per-stage exit criteria live in
[`tradingagents-migration-plan.md`](tradingagents-migration-plan.md). This table
is the shape; that document is the schedule.

---

## 14. Carried forward from the ORB+GEX engine

**This is a separate repository, and these arrive as copies, not imports.** The
ORB+GEX engine stays frozen and untouched — it is a working paper-trading path
and nothing here is worth risking it for. The cost of copying is that a fix to
`assert_paper_account` has to be applied twice; that is accepted deliberately,
and the two copies are expected to diverge. See the carry list in the migration
plan for what changes on the way across.

Worth lifting rather than rewriting — each cost real debugging:

| Module | Why |
|---|---|
| `execution/safety.py` | `assert_paper_account` — the DU/DF gate, called twice |
| `execution/confirmation.py` | Type-the-ticker gate, whatIf numbers in the prompt |
| `execution/journal.py` | Append-only JSONL that never raises |
| `logging_setup.py` | Benign-IB-code filter — otherwise 10091 fires once per contract at ERROR and buries real failures |

**IB connection lessons, all learned the hard way:**

- **`endDateTime` must be timezone-aware.** A naive value is resolved by IB in a
  zone of its own choosing — measured: naive `10:00` returned the 10:30–10:59
  bars, aware `10:00 ET` returned 09:30–09:59.
- **`TIME_ZONE` on the Gateway container must match the exchange.** The image
  defaults to `Etc/UTC` and IB then stamps bars in UTC.
- **Option chains: select by `tradingClass`, not exchange.**
  `reqSecDefOptParams` returns one entry per exchange × tradingClass; for SPY
  that is ~20 entries and all but one are a 3-strike mini class.
- **`reqMarketDataType(3)`** gives delayed greeks with no subscription. Error
  10091 is a substitution notice, not a failure.
- **Paper accounts need their own credentials**, and the paper account must be
  *created* in Client Portal → Account Configuration first. Client Portal will
  accept live credentials while the Gateway rejects them.

---

## 15. Risks and things that are bad ideas

1. **Look-ahead bias will silently invalidate the backtest.** Providers that
   cannot do point-in-time must **raise** in replay, not degrade quietly.
2. **Multi-agent debate manufactures confident consensus.** Three agents
   agreeing is not three pieces of evidence — they read the same reports. B4 is
   the only defense.
3. **Never let an LLM do arithmetic.**
4. **Do not build a generic graph engine.** This is why LangGraph is used rather
   than written (§1). The failure mode this replaces — spending a week building
   a worse LangGraph — is real; the new one to watch for is *using* more of
   LangGraph than orchestration. If a PR adds `langchain-core` chat models,
   `ToolNode`, or an `interrupt()`-based gate, that is the drift.
5. **LangGraph version churn is a live risk, not a hypothetical.** Pin exact
   versions, upgrade on purpose, and keep the node bodies free of framework
   types so an upgrade is a plumbing change rather than a rewrite.
6. **Survivorship bias will inflate the Stage 8 backtest** and free data cannot
   fix it (§7.5). Report it next to the numbers.
7. **Do not vendor a vector DB.** SQLite + numpy dot product over a few thousand
   reflection rows is exact, instant, and one fewer container.
8. **Do not schedule `execute`.** `decide` on a cron is fine; `execute` must be
   human-initiated, always.
9. **Scope risk.** 14 nodes × prompts × 6 providers is a lot of surface. Build
   the Stage 3 vertical slice before anything else.
10. **GEX cannot be backtested from IB** — expired contracts are removed from the
    chain, so historical option greeks and open interest are unavailable. If
    options ever enter the agent path, the evaluation harness needs a different
    data source.
11. **Copied code drifts.** §14's modules arrive as copies in a separate repo. A
    security-relevant fix to `assert_paper_account` must be applied in both
    places, and nothing will remind you.
12. Personal software on a paper account. Keep the human gate even after Stage 8
    gives reason to trust it.
