"""The ORB+GEX exclusivity guard. **One IB username, one Gateway session.**

This guard had no tests, and it broke in the way an untested guard breaks: not
loudly, but by being asked a question it could not answer and giving an answer
anyway.

``make gateway-start`` ran it through ``$(RUN)``, which means inside the ``dev``
container. There it has no Docker CLI and no socket, so the container check
returns ``None`` -- and its *port* check does not fail at all, because
``127.0.0.1:4002`` in a container is the container's own loopback. One check
abstains and the other confidently measures the wrong machine.

The consequences were a visible failure and a silent one, and the silent one is
the reason these tests exist:

* ``make gateway-start`` exited 2 and died, with compose noise on screen and a
  message that said "could not ask Docker" without saying why not.
* ``desk doctor`` mapped exit 2 onto "no conflict", so running in the container
  -- the default -- it could never report an ORB conflict at all.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_gateway_exclusive.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("check_gateway_exclusive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_gateway_exclusive"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The three outcomes, and that they stay three
# --------------------------------------------------------------------------- #


def test_no_container_and_no_port_is_clear(guard, monkeypatch) -> None:
    monkeypatch.setattr(guard, "_container_running", lambda name: False)
    monkeypatch.setattr(guard, "_port_listening", lambda port, host="127.0.0.1": False)
    code, lines = guard.check()
    assert code == 0
    assert "no conflict" in lines[0]


@pytest.mark.parametrize(
    ("container", "port", "expected_in_message"),
    [
        (True, False, "is running"),
        (False, True, "listening on"),
        (True, True, "is running"),
    ],
)
def test_either_signal_alone_is_conclusive(
    guard, monkeypatch, container: bool, port: bool, expected_in_message: str
) -> None:
    """*"Neither alone is sufficient; either alone is conclusive."*

    The container check misses a Gateway started outside Docker or under
    another name; the port check misses a container whose ports are not
    published. So either one firing is a conflict.
    """
    monkeypatch.setattr(guard, "_container_running", lambda name: container)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": port)
    code, lines = guard.check()
    assert code == 1
    assert "CONFLICT" in lines[0]
    assert any(expected_in_message in line for line in lines)
    # The fix has to be actionable, because this is read mid-frustration.
    assert any("docker stop ajj-ib-gateway" in line for line in lines)


def test_undeterminable_is_exit_2_not_exit_0(guard, monkeypatch) -> None:
    """**The distinction the whole fix rests on.**

    "No conflict" and "could not tell" must never collapse into one code. A
    guard protecting a single-session credential has to fail closed.
    """
    monkeypatch.setattr(guard, "_container_running", lambda name: None)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": False)
    monkeypatch.setattr(guard, "in_container", lambda: False)
    code, _ = guard.check()
    assert code == 2


# --------------------------------------------------------------------------- #
# The message names the real cause
# --------------------------------------------------------------------------- #


def test_inside_a_container_the_message_says_so(guard, monkeypatch) -> None:
    """The original text was true and useless.

    "Could not ask Docker whether the ORB+GEX Gateway is running" does not say
    that the reason is where you are standing, and the compose output above it
    on screen was about something else entirely.
    """
    monkeypatch.setattr(guard, "_container_running", lambda name: None)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": False)
    monkeypatch.setattr(guard, "in_container", lambda: True)
    code, lines = guard.check()
    text = "\n".join(lines)

    assert code == 2
    assert "cannot run inside a container" in text
    # It must say WHY the port probe is not reassurance, which is the subtler
    # half: that check does not abstain, it answers about the wrong loopback.
    assert "loopback" in text
    # And how to actually run it.
    assert "make check-gateway" in text


def test_outside_a_container_the_message_blames_docker_not_the_container(
    guard, monkeypatch
) -> None:
    monkeypatch.setattr(guard, "_container_running", lambda name: None)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": False)
    monkeypatch.setattr(guard, "in_container", lambda: False)
    _, lines = guard.check()
    text = "\n".join(lines)
    assert "Docker is not on PATH" in text
    assert "cannot run inside a container" not in text


def test_in_container_detection_is_false_on_this_host(guard) -> None:
    """The suite runs on the host, so the detector must say so.

    Guards against a detector that returns True everywhere, which would make
    the container branch above fire for real users on their own machines.
    """
    assert guard.in_container() is False


def test_in_container_detection_reads_dockerenv(guard, monkeypatch, tmp_path) -> None:
    marker = tmp_path / ".dockerenv"
    marker.write_text("")
    real_path = guard.Path

    class FakePath:
        def __init__(self, value: str) -> None:
            self._value = value

        def exists(self) -> bool:
            return self._value == "/.dockerenv"

        def read_text(self) -> str:
            raise OSError("no procfs")

    monkeypatch.setattr(guard, "Path", FakePath)
    try:
        assert guard.in_container() is True
    finally:
        monkeypatch.setattr(guard, "Path", real_path)


# --------------------------------------------------------------------------- #
# It really runs standalone, which is the point of it being stdlib-only
# --------------------------------------------------------------------------- #


def test_the_script_runs_with_no_project_environment() -> None:
    """*"Deliberately stdlib-only... so it runs from a Makefile before any
    environment exists."*

    Run as a subprocess with no cwd inside the repo and nothing importable, so
    an accidental ``from research_desk import ...`` would fail here rather
    than on somebody's first ``make gateway-start``.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=60, cwd="/",
    )
    assert result.returncode in (0, 1, 2), result.stderr
    assert "Traceback" not in result.stderr


def test_quiet_prints_nothing_when_clear(guard, monkeypatch, capsys) -> None:
    monkeypatch.setattr(guard, "_container_running", lambda name: False)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": False)
    assert guard.main(["--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_conflict_prints_to_stderr_even_when_quiet(guard, monkeypatch, capsys) -> None:
    """``--quiet`` suppresses the all-clear, never the conflict."""
    monkeypatch.setattr(guard, "_container_running", lambda name: True)
    monkeypatch.setattr(guard, "_port_listening", lambda p, host="127.0.0.1": False)
    assert guard.main(["--quiet"]) == 1
    captured = capsys.readouterr()
    assert "CONFLICT" in captured.err
    assert captured.out == ""


# --------------------------------------------------------------------------- #
# The Makefile must not route these through a container again
# --------------------------------------------------------------------------- #


def test_the_makefile_runs_the_guard_on_the_host() -> None:
    """The regression that started this. Asserted on the Makefile text.

    ``$(RUN)`` is the container by default. Both of the guard's checks read
    host state, so routing it through ``$(RUN)`` makes it structurally unable
    to answer -- which is how ``make gateway-start`` broke.
    """
    makefile = (REPO_ROOT / "Makefile").read_text()
    for line in makefile.splitlines():
        if "check_gateway_exclusive.py" not in line or line.lstrip().startswith("#"):
            continue
        assert "$(HOST_RUN)" in line, (
            f"{line.strip()!r} does not use $(HOST_RUN). The guard reads the "
            "host's Docker daemon and the host's loopback; in a container it "
            "cannot see either, and its port probe silently answers about the "
            "container's own loopback instead."
        )


def test_gateway_start_does_not_swallow_the_guards_exit_code() -> None:
    """``check-gateway`` is a report and may use ``|| true``. ``gateway-start``
    is a gate and must not: a guard that cannot see has to stop the start."""
    makefile = (REPO_ROOT / "Makefile").read_text()
    body = makefile.split("gateway-start:")[1].split("\n.PHONY")[0]
    guard_lines = [
        line for line in body.splitlines()
        if "check_gateway_exclusive.py" in line and not line.lstrip().startswith("@#")
    ]
    assert guard_lines, "gateway-start no longer runs the exclusivity guard"
    for line in guard_lines:
        assert "|| true" not in line, (
            "gateway-start swallows the guard's exit code. One IB username "
            "supports one Gateway session; an unanswerable check must refuse "
            "to start, not shrug."
        )


# --------------------------------------------------------------------------- #
# doctor's three-way read
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, None), (2, "unknown"), (127, "unknown")],
)
def test_doctor_separates_no_conflict_from_could_not_tell(
    monkeypatch, returncode: int, expected: str | None
) -> None:
    """The silent fail-open. Exit 2 used to come back as ``None``, i.e. "clear".

    Inside the container exit 2 is the *only* possible outcome, so `desk
    doctor` -- which runs in the container by default -- could never report an
    ORB conflict and claimed the Gateway was merely unreachable.
    """
    from research_desk import cli

    class Result:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stderr = ""
            self.stdout = ""

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Result())
    got = cli._orb_gateway_detected()

    if expected is None:
        assert got is None
    else:
        assert got == cli.ORB_UNKNOWN


def test_doctor_reports_the_conflict_description(monkeypatch) -> None:
    from research_desk import cli

    class Result:
        returncode = 1
        stdout = ""
        stderr = "CONFLICT: ...\n\n  - container 'ajj-ib-gateway' is running\n"

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Result())
    assert cli._orb_gateway_detected() == "container 'ajj-ib-gateway' is running"
