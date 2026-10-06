"""Content-addressed JSON cache. Built before the second provider, not the sixth.

Migration plan, Stage 2 traps: *"Build the cache before the second provider,
not after the sixth."* It is here first because three separate requirements all
turn out to be the same mechanism:

* **Point-in-time replay (§7.4).** In ``mode="replay"`` the cache is the *only*
  permitted source; HTTP is hard-disabled. That is only possible if every fetch
  went through here from the start.
* **Rate-limit survival.** EDGAR is 10 req/s, Finnhub 60/min, GDELT one per 5 s.
  A 15-symbol run that re-fetches on every iteration is unusable during prompt
  development.
* **The IB bridge.** ``decide`` may not import ``ib_async`` (§0), so bars reach
  it *only* as a cache artefact written by the ``execute`` side. The cache is
  the file in "two processes that communicate through a file".

The key is ``sha256(provider, method, sorted(kwargs), as_of)``, per §7.4.
``as_of=None`` means "not pinned to a date" and is used for append-only series
like daily bars, where the honest unit is one long history sliced at read time
rather than a separate copy per decision date.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheMiss(LookupError):
    """Asked for something not cached, in a mode that forbids fetching."""


def cache_key(provider: str, method: str, as_of: date | None, kwargs: dict[str, Any]) -> str:
    """Stable across runs, process restarts and key order."""
    blob = json.dumps(
        {
            "provider": provider,
            "method": method,
            "as_of": as_of.isoformat() if as_of else None,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


class Cache:
    """One JSON file per entry, under ``<root>/<provider>/<key>.json``.

    Deliberately a directory of files rather than SQLite: the entries are
    inspectable with ``cat`` while debugging a provider, which during Stage 2
    is most of the work. Revisit if the entry count ever makes that a problem.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(
        self, provider: str, method: str, as_of: date | None = None, **kwargs: Any
    ) -> Path:
        key = cache_key(provider, method, as_of, kwargs)
        return self.root / provider / f"{method}-{key}.json"

    def get(
        self, provider: str, method: str, as_of: date | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        path = self.path_for(provider, method, as_of, **kwargs)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
        except json.JSONDecodeError:
            # A truncated write must not be mistaken for a bad upstream reply.
            logger.warning("Discarding unreadable cache entry %s", path)
            return None
        return entry

    def put(
        self,
        provider: str,
        method: str,
        payload: Any,
        as_of: date | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write atomically. A partial entry is worse than a miss."""
        path = self.path_for(provider, method, as_of, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "provider": provider,
            "method": method,
            "as_of": as_of.isoformat() if as_of else None,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payload": payload,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True, default=str))
        tmp.replace(path)
        return path

    def entries(self, provider: str | None = None) -> list[Path]:
        where = self.root / provider if provider else self.root
        return sorted(where.rglob("*.json")) if where.exists() else []
