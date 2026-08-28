"""Data fetching. **Stage 2** -- the largest stage, and the one most likely to sprawl.

Four rules, all of which are cheap now and expensive to retrofit:

1. **Everything goes through ``registry.py``.** Nothing imports a provider module
   directly. This is what makes the prefetch functions promotable to real tools
   later without touching their call sites (architecture §2).
2. **Tool-shaped signatures.** Fully typed parameters, no ``**kwargs``,
   JSON-serialisable returns, and a docstring written as if it were already the
   tool description a model would read -- so the schema can be *generated* from
   the signature rather than hand-written twice.
3. **``as_of`` on every method, always.** The point-in-time guard lives in the
   function, not in the caller, because a tool a model can call at an arbitrary
   moment is exactly where look-ahead bias creeps back in.
4. **Every provider declares ``supports_point_in_time``.** In ``mode="replay"``
   the cache is the only permitted source: HTTP is hard-disabled and a provider
   that cannot do point-in-time **raises** rather than degrading quietly.
   Look-ahead bias is the number one way agentic backtests produce fictional
   Sharpe ratios (architecture §7.4, §15.1).

Build the cache before the *second* provider, not after the sixth.

Planned: ``stooq``, ``ib``, ``edgar``, ``finnhub``, ``finra``, ``fred``,
``gdelt``, ``cboe``, plus ``cache`` and ``limiter``.

Note that ``scripts/gdelt_collect.py`` deliberately does **not** live here. It is
a standalone collector, decoupled from everything, because it has to start
accumulating before this package exists -- see its docstring.
"""
