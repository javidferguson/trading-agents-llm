"""News tone and coverage volume -- **read from the collector's cache only.**

``scripts/gdelt_collect.py`` writes; this reads. The split is deliberate and
not merely tidy:

* **GDELT's window closes.** The open API serves a rolling ~3 months, so tone
  for a date must be captured near that date or it is gone (§7.5). The
  collector runs on its own schedule, independent of any research run.
* **The rate limit makes it unusable inline.** One request per 5 seconds,
  stated only in the body of its 429, means a 15-symbol fetch takes 2.5
  minutes. No node can wait for that.
* **So in every mode this path is cache-only**, which makes it point-in-time by
  construction -- the same property the IB bars bridge has, for the same
  reason.

An empty cache is therefore an *expected* state with a specific meaning, and
the gaps say so: it reports "not collected" rather than a neutral zero.
§7.5 is emphatic on that point -- *"never zero, and never a neutral default. A
silent zero is indistinguishable from genuinely neutral coverage, and it will
quietly corrupt the ablation that decides whether sentiment earns its place."*
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from .base import ProviderSpec

logger = logging.getLogger(__name__)

SPEC = ProviderSpec(
    name="gdelt",
    #: Only within the window the collector captured -- which is exactly why
    #: the collector exists and why it cannot wait.
    supports_point_in_time=True,
    min_interval_s=0.0,
    reads_network=False,   # the collector owns the network; this reads files
    role="news tone and coverage volume, read from the collector's cache",
)


class GdeltProvider:
    """Reads the per-symbol JSON stores the collector writes."""

    spec = SPEC

    def __init__(self, cache_dir: str | Path):
        #: The collector writes to data/cache/gdelt/<SYMBOL>.json directly
        #: rather than through providers/cache.py -- it predates the provider
        #: layer on purpose (§7.5's build-order note) and must stay runnable
        #: with nothing but PyYAML.
        self.root = Path(cache_dir) / "gdelt"

    def store_path(self, symbol: str) -> Path:
        return self.root / f"{symbol.upper()}.json"

    async def news_tone(self, symbol: str, as_of: date) -> list[dict[str, Any]]:
        """Daily tone and article counts up to ``as_of``, oldest first.

        Returns ``[]`` when nothing has been collected for the symbol. The
        caller must distinguish that from "collected, and there was no
        coverage" -- the collector records the latter explicitly with
        ``observed: true``.
        """
        path = self.store_path(symbol)
        if not path.exists():
            return []
        try:
            store = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("unreadable GDELT store %s", path)
            return []

        cutoff = as_of.isoformat()
        out = []
        for day, record in sorted((store.get("days") or {}).items()):
            if day > cutoff:
                continue
            out.append({
                "day": date.fromisoformat(day),
                "tone": record.get("tone"),
                "articles": record.get("articles"),
                "observed": record.get("observed", False),
            })
        return out

    def collected_symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))
