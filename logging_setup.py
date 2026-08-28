"""Logging configuration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "orb_gamma.log"

# IB codes that are informational despite arriving on the error channel.
BENIGN_IB_CODES = {
    10091,  # delayed market data substituted for live (once per contract)
    10167,  # ditto, alternate phrasing
    2104,   # market data farm connection is OK
    2106,   # historical data farm connection is OK
    2107,   # historical data farm is inactive but should be available on demand
    2108,   # market data farm connection is inactive but should be available
    2119,   # market data farm is connecting
    2158,   # security definition server connection is OK
}


class _BenignIBCodeFilter(logging.Filter):
    """Demote IB's routine notifications from ERROR to DEBUG.

    Returning False would hide them entirely; they are occasionally useful when
    diagnosing a data problem, so keep them at DEBUG instead.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.levelno >= logging.ERROR and message.startswith("Error "):
            for code in BENIGN_IB_CODES:
                if message.startswith(f"Error {code},"):
                    record.levelno = logging.DEBUG
                    record.levelname = "DEBUG"
                    return logging.getLogger().isEnabledFor(logging.DEBUG)
        return True


def setup_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE))
    except OSError:
        # A read-only mount should not stop the engine from running.
        print(f"WARNING: could not open {LOG_FILE}; logging to stdout only", file=sys.stderr)

    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # ib_async is extremely chatty at DEBUG and drowns the engine's own output.
    logging.getLogger("ib_async").setLevel(logging.WARNING)

    # IB reports a batch of routine notifications through the same error channel
    # as real failures, and ib_async logs them all at ERROR. The noisiest by far
    # is 10091, emitted once per contract: "requires additional subscription...
    # Delayed market data is available." That is a substitution notice, not a
    # failure -- measured against a paper account, gamma and open interest
    # arrive for every contract regardless, and no combination of market data
    # type or generic tick list suppresses it. Left at ERROR it buries the
    # failures that matter, so demote the known-benign codes to DEBUG.
    logging.getLogger("ib_async.wrapper").addFilter(_BenignIBCodeFilter())
