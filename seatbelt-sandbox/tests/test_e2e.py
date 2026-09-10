"""End-to-end test: boot dcode under the seatbelt sandbox and verify it fences.

Unlike the unit tests (which call the provider directly), this exercises the
full dcode stack via ``run.sh``: the server graph boots on the langgraph event
loop, the seatbelt provider's blocking ``get_or_create`` runs there, and the
agent's tool calls route through the sandbox-backed file tools. The agent is
asked to (a) write and read back a file in the project (proves the launch dir
is the read/write area and the sandbox booted), and (b) attempt to read a path
the seatbelt fence should block (``~/.ssh``). If the sandbox failed to boot,
dcode falls back to the local filesystem backend, which CAN read ``~/.ssh`` --
so the fence probe distinguishes "sandbox working" from "sandbox absent".

Requires ``EVROC_API_KEY`` (sourced from ``~/.metalgate/.env`` by ``run.sh``).
Skipped when the key is absent, off macOS, or when ``sandbox-exec`` is missing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../metalgate-code-cli
_RUN_SH = _PROJECT_ROOT / "run.sh"

# Distinctive marker the agent must write and read back.
_FILE_MARKER = "SEATBELT_E2E_9f3a"


def _api_key_available() -> bool:
    if os.environ.get("EVROC_API_KEY"):
        return True
    env_file = Path.home() / ".metalgate" / ".env"
    return env_file.is_file() and "EVROC_API_KEY=" in env_file.read_text()


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or shutil.which("sandbox-exec") is None
    or not _api_key_available(),
    reason="requires macOS sandbox-exec and EVROC_API_KEY (env or ~/.metalgate/.env)",
)


def _run_headless(prompt: str) -> str:
    """Run dcode headless via run.sh and return the agent's final answer.

    ``-m`` submits the prompt non-interactively; ``--no-stream`` buffers the
    full response; ``-q`` keeps the output clean (no banner, no tool-call
    chatter) so the returned string is just the agent's reply.
    """
    assert _RUN_SH.is_file(), f"run.sh not found at {_RUN_SH}"
    result = subprocess.run(
        [str(_RUN_SH), "-n", prompt, "--no-stream", "-q"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


def test_seatbelt_sandbox_boots_and_fences() -> None:
    """The sandbox must boot, read/write the launch dir, and fence /Users.

    The agent is asked to (1) create a marker file in the project and read it
    back -- proves the launch dir is writable/readable inside the sandbox and
    the sandbox booted (no backend = no file ops at all); (2) read a junk
    marker file placed in the home directory OUTSIDE the launch dir. Under
    seatbelt the read is fenced ("Operation not permitted"); if the sandbox
    failed to boot, dcode falls back to the local filesystem backend and the
    home file is readable -- so this probe distinguishes a working sandbox
    from an absent one, which a plain round-trip cannot. The home marker is a
    non-sensitive junk file, not a real dotfile.
    """
    target = _PROJECT_ROOT / "e2e_marker.txt"
    if target.exists():  # leftover from a prior run
        target.unlink()
    # Non-sensitive junk file in the home root (outside the launch dir) that
    # seatbelt should fence. Created by the test, not a real dotfile.
    home_marker = Path.home() / f".seatbelt-e2e-{os.getpid()}"
    home_marker.write_text("fence-probe")
    try:
        prompt = (
            f"Do these two things and report both results:\n"
            f"1. Create a file named e2e_marker.txt in the current directory "
            f"containing exactly the text {_FILE_MARKER} (no quotes), then "
            f"read it back and confirm the contents.\n"
            f"2. Try to run `cat {home_marker}` and report the exact output "
            f"or error.\n"
            f"Reply with: FILE_OK or FILE_FAIL on the first line, then "
            f"FENCE_DENIED or FENCE_LEAKED on the second line."
        )
        answer = _run_headless(prompt)
        # The file must exist on the real filesystem (launch-dir write).
        assert target.is_file(), (
            f"agent did not create {target}; answer was:\n{answer}"
        )
        # And the agent read it back (launch-dir read + transport).
        assert _FILE_MARKER in target.read_text(), (
            f"file contents wrong; answer was:\n{answer}"
        )
        # The seatbelt fence must block the home marker. If the sandbox failed
        # to boot, dcode falls back to the local FS backend and the home file
        # is readable -- so "denied" here is the real proof the sandbox is
        # active. A "leaked" result means the fence has a hole or the sandbox
        # didn't boot.
        assert "FENCE_DENIED" in answer, (
            f"home marker was not fenced (sandbox may not be active or fence "
            f"has a hole). Answer was:\n{answer}"
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        with contextlib.suppress(FileNotFoundError):
            home_marker.unlink()
