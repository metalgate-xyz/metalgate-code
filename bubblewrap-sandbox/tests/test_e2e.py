"""End-to-end: boot dcode under the bubblewrap sandbox and verify it fences.

Exercises the full dcode stack via ``run.sh``: the server graph boots, the
bubblewrap provider's blocking ``get_or_create`` runs on the langgraph event
loop, and the agent's tool calls route through the sandbox-backed file tools.
The agent writes+reads a file in the project (proves the launch dir is the
read/write area and the sandbox booted) and attempts to read a file the
bubblewrap fence should block.

Requires ``EVROC_API_KEY`` (sourced from ``~/.metalgate/.env`` by ``run.sh``).
Skipped when the key is absent, off Linux, or when ``bwrap`` is missing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../metalgate-code-cli
_RUN_SH = _PROJECT_ROOT / "run.sh"

# Distinctive marker the agent must write and read back.
_FILE_MARKER = "BUBBLEWRAP_E2E_9f3a"


def _api_key_available() -> bool:
    if os.environ.get("EVROC_API_KEY"):
        return True
    env_file = Path.home() / ".metalgate" / ".env"
    return env_file.is_file() and "EVROC_API_KEY=" in env_file.read_text()


pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bwrap") is None
    or not _api_key_available(),
    reason="requires Linux bwrap and EVROC_API_KEY (env or ~/.metalgate/.env)",
)


def _run_headless(prompt: str) -> str:
    """Run dcode headless via run.sh and return the agent's final answer."""
    assert _RUN_SH.is_file(), f"run.sh not found at {_RUN_SH}"
    env = dict(os.environ)
    env["DCODE_SANDBOX"] = "bubblewrap"
    result = subprocess.run(
        [str(_RUN_SH), "-n", prompt, "--no-stream", "-q", "--sandbox", "bubblewrap"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=env,
    )
    return (result.stdout + result.stderr).strip()


def test_bubblewrap_sandbox_boots_and_fences() -> None:
    """The sandbox must boot, read/write the launch dir, and fence $HOME.

    The agent (1) creates a marker file in the project and reads it back:
    proves the launch dir is writable/readable and the sandbox booted; (2)
    tries to read a junk marker placed under ``$HOME`` outside the launch dir,
    which bubblewrap must block (the home path is not mounted).
    """
    target = _PROJECT_ROOT / "e2e_marker.txt"
    if target.exists():  # leftover from a prior run
        target.unlink()
    # Throwaway directory under $HOME (outside the launch dir) to probe the fence.
    outside_dir = (
        Path.home().resolve() / f".dcode-sandbox-e2e-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    try:
        outside_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        # $HOME may be read-only when pytest itself runs sandboxed.
        pytest.skip(f"cannot create ad hoc dir under $HOME: {exc}")
    home_marker = outside_dir / "fence-probe.txt"
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
        assert target.is_file(), f"agent did not create {target}; answer was:\n{answer}"
        # And the agent read it back (launch-dir read + transport).
        assert _FILE_MARKER in target.read_text(), (
            f"file contents wrong; answer was:\n{answer}"
        )
        # The bubblewrap fence must block the home marker.
        assert "FENCE_DENIED" in answer, (
            f"home marker was not fenced. Answer was:\n{answer}"
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        shutil.rmtree(outside_dir, ignore_errors=True)
