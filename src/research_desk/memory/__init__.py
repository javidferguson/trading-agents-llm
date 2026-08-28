"""Reflection storage and retrieval, keyed by ``(symbol, regime)``. **Stage 9.**

> **Do not vendor a vector DB** (architecture §15.7). SQLite plus a numpy dot
> product over a few thousand reflection rows is exact, instant, and one fewer
> container.

The exit gate is not "a retrieval happened". It is that reflections are
retrieved **and measurably change decisions** -- which means an ablation with
``memory.enabled=false``, not a log line.
"""
