"""A deterministic book for the Stage 6 tests. **Not ``data/portfolio.yaml``.**

That file is **state**, not a fixture. It holds whatever the account holds, and
at Stage 7 ``execute`` overwrites it from the broker -- which is exactly what its
own header has promised since Stage 6. The moment it did, twelve tests went red
for a reason that had nothing to do with what they were testing.

The same mistake, in a third place today: ``tests/graph/`` hardcoded a decision
date against a book that gets re-marked, two tests named GOOGL as "unheld" after
GOOGL joined the book, and these asserted against share counts in a file the
broker now owns.

**So the fixture is the seed RECIPE, not the seed file.**
``scripts/seed_portfolio.py:SEED_BOOK`` plus the bars cache is committed,
deterministic, and regenerates the same book every time -- and it is the single
definition, so the demo book a developer installs with ``make portfolio-seed``
and the book these tests use cannot drift apart.

Anything that genuinely wants the live account reads ``load_portfolio()``
directly and says why.

At the test-root rather than in one package: ``tests/graph/`` needs it too,
because the compliance NODE reads the book itself and must be given the fixture
rather than the account.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seeder():
    spec = importlib.util.spec_from_file_location(
        "seed_portfolio", REPO_ROOT / "scripts" / "seed_portfolio.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_portfolio"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def seeded_book():
    """The Stage 6 gate's book, rebuilt from ``SEED_BOOK`` and the bars cache.

    Properties the tests rely on, all by construction:

    * exactly ``max_positions`` positions, so the slot-blocked path is reachable
    * two symbols at ``max_position_pct``
    * TSM well under its target, so a BUY actually sizes
    * three themes over target and four under

    Skips rather than fails when the bars cache is absent: a fresh clone has no
    cache, and "run ``make bars``" is a setup instruction, not a defect.
    """
    seeder = _seeder()
    try:
        return seeder.seed()
    except SystemExit as exc:
        pytest.skip(f"no bars cache for the seeded book: {exc}")
