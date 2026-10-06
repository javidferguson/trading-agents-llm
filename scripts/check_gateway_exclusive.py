#!/usr/bin/env python3
"""Refuse to start this project's IB Gateway while the ORB+GEX engine's is up.

WHY THIS EXISTS
===============
This repo runs its own Gateway (migration plan §0, as revised). Docker makes
that trivial. IB does not: **one IB username supports one Gateway/TWS session.**
Both repos use the same credentials, so two Gateways means two logins to one
account and IB resolves it by evicting somebody.

Our container is configured ``EXISTING_SESSION_DETECTED_ACTION=secondary``, so
it steps aside rather than stealing the session. That is the right default but
it is a *late* defence: by then you have pulled an image, started a container,
and are reading IBC logs to work out why nothing is listening.

This is the early one. It costs 20ms and it says the actual reason.

WHAT IT CHECKS
==============
Either of these means the ORB engine's Gateway is up:

1. a running container named ``ajj-ib-gateway``
2. something listening on ``127.0.0.1:4002`` -- its published paper port

Two checks rather than one because they fail differently: the container check
misses a Gateway started outside Docker or under another name, and the port
check misses a container whose ports are not published. Neither alone is
sufficient; either alone is conclusive.

Deliberately stdlib-only and importing nothing from ``research_desk``, so it
runs from a Makefile before any environment exists.

Exit codes: 0 clear, 1 conflict, 2 could not tell.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys

#: The ORB+GEX engine's Gateway container.
OTHER_CONTAINER = "ajj-ib-gateway"

#: Its published paper port on the host. Ours is 4012, deliberately different.
OTHER_HOST_PORT = 4002

OUR_CONTAINER = "desk-ib-gateway"
OUR_HOST_PORT = 4012


def _container_running(name: str) -> bool | None:
    """True / False, or None when Docker cannot be consulted."""
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return name in result.stdout.split()


def _port_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def check() -> tuple[int, list[str]]:
    """Return ``(exit_code, lines)``."""
    container = _container_running(OTHER_CONTAINER)
    port = _port_listening(OTHER_HOST_PORT)

    if not container and not port:
        if container is None:
            return 2, [
                "Could not ask Docker whether the ORB+GEX Gateway is running.",
                f"Port {OTHER_HOST_PORT} is clear, but that alone is not proof.",
                "Check by hand before starting this project's Gateway.",
            ]
        return 0, [
            f"ok   no conflict: {OTHER_CONTAINER} is not running and "
            f"127.0.0.1:{OTHER_HOST_PORT} is free."
        ]

    found = []
    if container:
        found.append(f"container {OTHER_CONTAINER!r} is running")
    if port:
        found.append(f"something is listening on 127.0.0.1:{OTHER_HOST_PORT}")

    return 1, [
        "CONFLICT: the ORB+GEX engine's IB Gateway appears to be up.",
        "",
        *(f"  - {item}" for item in found),
        "",
        "Both projects use the same IB credentials, and one IB username",
        "supports one Gateway session. Starting this project's Gateway now",
        "would either be refused (we default to",
        "EXISTING_SESSION_DETECTED_ACTION=secondary) or, with that overridden,",
        "would evict a possibly-mid-position ORB trading session.",
        "",
        "Stop the other one first:",
        "",
        f"    docker stop {OTHER_CONTAINER}",
        "",
        "then re-run `make gateway-start`.",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true",
                        help="print only on conflict")
    args = parser.parse_args(argv)

    code, lines = check()
    if code != 0 or not args.quiet:
        stream = sys.stdout if code == 0 else sys.stderr
        print("\n".join(lines), file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
