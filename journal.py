"""Append-only trade journal.

The previous engine kept its trade log in a Python list that was discarded on
exit, which is why "comprehensive trade tracking" never produced a file. This
writes JSONL to disk at submit and on every subsequent status change.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = "data/journal"


def _encode(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


class Journal:
    """One JSONL file per session, appended to as events happen."""

    def __init__(self, directory: str | Path = DEFAULT_JOURNAL_DIR, session: str | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = session or datetime.now().strftime("%Y%m%d")
        self.path = self.directory / f"trades_{stamp}.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        """Append one event. Never raises -- a journal failure must not kill a trade."""
        record = {"timestamp": datetime.now().isoformat(), "event": event, **fields}
        try:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, default=_encode) + "\n")
        except OSError:
            logger.exception("Failed to write journal event %r to %s", event, self.path)
