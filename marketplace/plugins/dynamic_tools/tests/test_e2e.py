"""End-to-end tests for the dynamic_tools plugin.

Run real headless ``dcode`` sessions through ``run.sh`` -- which sets
``DEEPAGENTS_CODE_EXPERIMENTAL=1`` (the env var that gates the entire extension
runtime), installs the plugin, and passes args through to ``dcode`` -- and
asserts the model can call tools dynamically loaded from the local
``.metalgate/tools`` directory and the agent-repo ``.metalgate/global_tools``
directory.

These are real network+model tests: they require ``EVROC_API_KEY`` (sourced
from ``~/.metalgate/.env`` by ``run.sh``). They are slow and skipped when the
key is absent.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../metalgate-code-cli
_RUN_SH = _PROJECT_ROOT / "run.sh"
_TOOLS_DIR = _PROJECT_ROOT / ".metalgate" / "tools"
_GLOBAL_TOOLS_DIR = _PROJECT_ROOT / ".metalgate" / "global_tools"

LOCAL_TOOL = textwrap.dedent(
    """
    def echo_twice(text: str) -> str:
        '''Echo the given text twice, separated by a newline.'''
        return f"{text}\\n{text}"
    """
)

GLOBAL_TOOL = textwrap.dedent(
    """
    def shout(text: str) -> str:
        '''Return the text uppercased with a trailing exclamation.'''
        return f"{text.upper()}!"
    """
)


def _api_key_available() -> bool:
    if os.environ.get("EVROC_API_KEY"):
        return True
    env_file = Path.home() / ".metalgate" / ".env"
    return env_file.is_file() and "EVROC_API_KEY=" in env_file.read_text()


pytestmark = pytest.mark.skipif(
    not _api_key_available(),
    reason="EVROC_API_KEY not set (env or ~/.metalgate/.env); e2e needs a live model",
)


def _run_headless(prompt: str) -> str:
    """Run dcode headless via run.sh and return combined stdout+stderr."""
    assert _RUN_SH.is_file(), f"run.sh not found at {_RUN_SH}"
    result = subprocess.run(
        [str(_RUN_SH), "-n", prompt, "-q", "--no-stream"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.fixture()
def local_tool_file(tmp_path):
    """Write a sample tool into .metalgate/tools (local project dir). Clean up after."""
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tool_file = _TOOLS_DIR / "sample_tool.py"
    tool_file.write_text(LOCAL_TOOL)
    yield tool_file
    tool_file.unlink(missing_ok=True)


@pytest.fixture()
def global_tool_file(tmp_path):
    """Write a sample tool into .metalgate/global_tools (agent-repo global dir). Clean up after."""
    _GLOBAL_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tool_file = _GLOBAL_TOOLS_DIR / "global_tool.py"
    tool_file.write_text(GLOBAL_TOOL)
    yield tool_file
    tool_file.unlink(missing_ok=True)
    # Don't leave an empty global_tools dir behind either.
    if _GLOBAL_TOOLS_DIR.exists() and not any(_GLOBAL_TOOLS_DIR.iterdir()):
        _GLOBAL_TOOLS_DIR.rmdir()


def test_model_calls_local_tool(local_tool_file):
    """The model must see and invoke echo_twice from .metalgate/tools."""
    output = _run_headless(
        'Use the echo_twice tool with text "hi" and report its exact return value. '
        "Do not use any other tools."
    )
    # echo_twice returns "hi\nhi". The model may render it with literal "\n" or
    # as actual newlines; either way "hi" appears twice in the output.
    assert output.lower().count("hi") >= 2, (
        f"echo_twice output not found (hi appears <2 times). Output:\n{output}"
    )


def test_model_calls_global_tool(global_tool_file):
    """The model must see and invoke shout from .metalgate/global_tools."""
    output = _run_headless(
        'Use the shout tool with text "hey" and report its exact return value. '
        "Do not use any other tools."
    )
    assert "HEY!" in output, f"shout output 'HEY!' not found. Output:\n{output}"
