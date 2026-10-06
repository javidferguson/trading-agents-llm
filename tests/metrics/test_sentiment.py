"""Sentiment normalisation, and the silent-zero failure §7.5 is emphatic about.

> *"Any replay earlier than the cache start must return sentiment fields as
> None WITH A STATED REASON -- never zero, and never a neutral default. A
> silent zero is indistinguishable from genuinely neutral coverage, and it
> will quietly corrupt the ablation that decides whether sentiment earns its
> place."*

That is the whole test file, really.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research_desk.metrics.sentiment import sentiment_metrics

AS_OF = date(2026, 10, 6)


def day(offset: int, tone: float | None, articles: int | None = 10,
        observed: bool = True) -> dict:
    return {
        "day": AS_OF - timedelta(days=offset),
        "tone": tone,
        "articles": articles,
        "observed": observed,
    }


# --------------------------------------------------------------------------- #
# The empty cache -- an expected state with a specific meaning
# --------------------------------------------------------------------------- #


def test_an_uncollected_symbol_is_never_neutral() -> None:
    metrics = sentiment_metrics([], AS_OF)

    assert not metrics.available(), "nothing may be inferred from nothing"
    assert len(metrics.gaps) == 6
    for reason in metrics.gaps.values():
        assert "not collected" in reason or "no GDELT history" in reason
        # The reason has to be actionable: the window is still closing.
        assert "gdelt_collect" in reason


def test_none_is_also_never_neutral() -> None:
    assert not sentiment_metrics(None, AS_OF).available()


# --------------------------------------------------------------------------- #
# Observed-zero vs not-collected: the distinction that matters
# --------------------------------------------------------------------------- #


def test_a_day_with_no_coverage_is_not_averaged_in_as_zero_tone() -> None:
    """The collector records a genuinely quiet day as tone=None, articles=0,
    observed=True. Averaging that in as 0.0 would pull every window toward
    neutral and make a quiet symbol look deliberately neutral."""
    days = [
        day(1, tone=-3.0),
        day(2, tone=None, articles=0),   # observed, no coverage
        day(3, tone=-3.0),
    ]
    metrics = sentiment_metrics(days, AS_OF)
    # Mean of the two days that HAD tone, not of three with a zero.
    assert metrics.news_tone_7d == pytest.approx(-3.0)


def test_observed_zero_days_still_count_toward_article_volume() -> None:
    """Zero articles IS a measurement of attention, unlike zero tone."""
    days = [day(1, tone=None, articles=0), day(2, tone=1.0, articles=20)]
    assert sentiment_metrics(days, AS_OF).article_count_7d == pytest.approx(20)


# --------------------------------------------------------------------------- #
# Windows and percentiles
# --------------------------------------------------------------------------- #


def test_tone_windows_respect_their_spans() -> None:
    days = [day(0, tone=5.0), day(3, tone=1.0), day(20, tone=-9.0)]
    metrics = sentiment_metrics(days, AS_OF)

    assert metrics.news_tone_1d == pytest.approx(5.0)
    assert metrics.news_tone_7d == pytest.approx(3.0)        # 5 and 1
    assert metrics.news_tone_30d == pytest.approx((5 + 1 - 9) / 3)


def test_nothing_in_the_window_is_a_gap_not_a_zero() -> None:
    metrics = sentiment_metrics([day(40, tone=2.0)], AS_OF)
    assert metrics.news_tone_1d is None
    assert "1 day" in metrics.gaps["news_tone_1d"]
    assert metrics.news_tone_30d is None


def test_a_percentile_needs_real_history() -> None:
    """§7.1 wants percentiles precisely because raw tone is meaningless to a
    small model -- but a percentile over a handful of days is noise wearing a
    number's clothes, which is worse."""
    thin = [day(i, tone=float(i % 5)) for i in range(1, 30)]
    metrics = sentiment_metrics(thin, AS_OF)

    assert metrics.news_tone_percentile_1y is None
    assert "60+" in metrics.gaps["news_tone_percentile_1y"]


def test_a_percentile_computes_once_history_is_deep_enough() -> None:
    # 200 days of mildly negative tone, then a strongly positive week.
    history = [day(i, tone=-2.0) for i in range(8, 208)]
    recent = [day(i, tone=5.0) for i in range(1, 8)]
    metrics = sentiment_metrics(history + recent, AS_OF)

    assert metrics.news_tone_percentile_1y is not None
    assert metrics.news_tone_percentile_1y > 0.9, "an unusually positive week"


def test_coverage_volume_zscore_flags_an_attention_spike() -> None:
    """§7.1: an unusual spike in coverage VOLUME is a better reason to pay
    attention than the tone of that coverage."""
    import random
    rng = random.Random(0)
    baseline = [day(i, tone=0.0, articles=10 + rng.randint(-2, 2)) for i in range(8, 95)]
    spike = [day(i, tone=0.0, articles=90) for i in range(1, 8)]

    metrics = sentiment_metrics(baseline + spike, AS_OF)
    assert metrics.coverage_volume_zscore_90d is not None
    assert metrics.coverage_volume_zscore_90d > 20, "a 9x spike should be unmistakable"


def test_the_baseline_excludes_the_week_it_is_measuring() -> None:
    """Otherwise the comparison is self-referential and damps its own signal.

    Measured: with the spike inside the baseline, a 9x spike scored 3.4;
    excluded, it scores ~57. The damping is worst exactly when the signal is
    strongest, which is the opposite of useful.
    """
    import random
    rng = random.Random(0)
    baseline = [day(i, tone=0.0, articles=10 + rng.randint(-2, 2)) for i in range(8, 95)]
    spike = [day(i, tone=0.0, articles=90) for i in range(1, 8)]

    z = sentiment_metrics(baseline + spike, AS_OF).coverage_volume_zscore_90d
    assert z > 20, (
        "a self-referential baseline would land near 3; this must not regress"
    )


def test_a_perfectly_flat_baseline_is_a_gap_not_an_infinity() -> None:
    days = [day(i, tone=0.0, articles=10) for i in range(8, 95)]
    days += [day(i, tone=0.0, articles=10) for i in range(1, 8)]
    metrics = sentiment_metrics(days, AS_OF)
    assert metrics.coverage_volume_zscore_90d is None


def test_a_thin_baseline_refuses_a_zscore() -> None:
    days = [day(i, tone=0.0, articles=10) for i in range(1, 20)]
    metrics = sentiment_metrics(days, AS_OF)
    assert metrics.coverage_volume_zscore_90d is None
    assert "30+" in metrics.gaps["coverage_volume_zscore_90d"]


def test_future_days_are_invisible() -> None:
    """The collector may hold days past as_of during a replay."""
    days = [day(-5, tone=99.0), day(1, tone=1.0)]
    metrics = sentiment_metrics(days, AS_OF)
    assert metrics.news_tone_1d == pytest.approx(1.0)
    assert metrics.news_tone_7d == pytest.approx(1.0), "99.0 is in the future"
