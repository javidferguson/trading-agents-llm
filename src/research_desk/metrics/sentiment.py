"""News tone and attention, normalised to percentiles (§7.1).

The normalisation is the point. A raw GDELT tone of ``-1.7`` is meaningless to
an 8B model and a percentile of its own trailing year is not -- and §7.1 is
explicit that the prompt must get the number and the direction, never an
adjective, because the short-horizon sign is contrarian and a small model told
"sentiment is very positive" will say BUY every time.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, timedelta
from typing import Any

from ..models.market import SentimentMetrics
from .indicators import _Block, percentile_rank

logger = logging.getLogger(__name__)

NOT_COLLECTED = (
    "no GDELT history collected for this symbol. Run "
    "`python scripts/gdelt_collect.py --days 90`; the API serves only a "
    "rolling ~3 months, so days not collected are unrecoverable"
)


def sentiment_metrics(
    tone_days: list[dict[str, Any]] | None,
    as_of: date,
) -> SentimentMetrics:
    """Tone and coverage volume from the collector's cache.

    An empty cache is an expected state with a specific meaning, so it reports
    "not collected" rather than a neutral zero -- §7.5: *"a silent zero is
    indistinguishable from genuinely neutral coverage, and it will quietly
    corrupt the ablation that decides whether sentiment earns its place."*
    """
    if not tone_days:
        return SentimentMetrics.unavailable(NOT_COLLECTED)

    block = _Block(len(tone_days))
    by_day = {record["day"]: record for record in tone_days}

    def window(days: int) -> list[dict[str, Any]]:
        start = as_of - timedelta(days=days)
        return [r for d, r in by_day.items() if start <= d <= as_of]

    def mean_tone(days: int) -> float | None:
        # Only days with a tone. A day the collector observed as zero-coverage
        # has tone None by design, and averaging it in as 0.0 would pull every
        # window toward neutral.
        values = [r["tone"] for r in window(days) if r.get("tone") is not None]
        return statistics.mean(values) if values else None

    for label, span in (("1d", 1), ("7d", 7), ("30d", 30)):
        block.compute(
            f"news_tone_{label}", 1, lambda s=span: mean_tone(s),
            note=f"no article tone recorded in the {span} day(s) before as_of",
        )

    def _tone_percentile() -> float | None:
        """Where the 7-day tone sits in this symbol's own trailing year."""
        current = mean_tone(7)
        if current is None:
            return None
        year_start = as_of - timedelta(days=365)
        history = [
            r["tone"] for d, r in by_day.items()
            if year_start <= d <= as_of and r.get("tone") is not None
        ]
        # A percentile over a handful of days is noise wearing a number's
        # clothes. 60 is the floor the shadow-mode corpus (§11) assumes.
        if len(history) < 60:
            return None
        return percentile_rank(history, current)
    block.compute(
        "news_tone_percentile_1y", 1, _tone_percentile,
        note="needs 60+ days of collected tone to rank against; "
             "percentiles over less are noise",
    )

    def _article_count() -> float | None:
        counts = [r["articles"] for r in window(7) if r.get("articles") is not None]
        return float(sum(counts)) if counts else None
    block.compute("article_count_7d", 1, _article_count,
                  note="no observed days in the 7 before as_of")

    def _volume_z() -> float | None:
        """Coverage volume against the symbol's own PRIOR 90-day baseline.

        §7.1: an unusual spike in coverage VOLUME is a better reason to pay
        attention than the tone of that coverage.

        **The baseline excludes the week being measured**, and that is not a
        detail. Including it makes the comparison self-referential: the spike
        inflates the standard deviation it is being divided by, so a larger
        spike partially hides itself. Measured on a synthetic 9x spike, the
        self-referential form scored 3.4 where the excluded form scores ~57 --
        the damping is severe, and it is worst exactly when the signal is
        strongest.
        """
        recent = [r["articles"] for r in window(7) if r.get("articles") is not None]

        baseline_start = as_of - timedelta(days=90)
        recent_start = as_of - timedelta(days=7)
        baseline = [
            r["articles"] for d, r in by_day.items()
            if baseline_start <= d < recent_start and r.get("articles") is not None
        ]
        if not recent or len(baseline) < 30:
            return None
        spread = statistics.stdev(baseline)
        if spread == 0:
            # Perfectly flat coverage. A z-score would be infinite, and "the
            # baseline never varies" is not the same as "a huge spike".
            return None
        return (statistics.mean(recent) - statistics.mean(baseline)) / spread
    block.compute("coverage_volume_zscore_90d", 1, _volume_z,
                  note="needs 30+ observed days in the 90 BEFORE the last "
                       "week to form a baseline")

    return block.build(SentimentMetrics)
