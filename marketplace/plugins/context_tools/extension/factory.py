"""dcode extension factory for the contextual symbol search tools.

Wires the language-specific tracer into the dcode extension registrar and
exposes the six code-navigation tools to the model.

Runs entirely host-side: the language servers (``ty`` for Python,
``gopls`` for Go) are launched as local subprocesses against the project
root (``api.cwd``), and file access is direct on the host filesystem.
No sandbox backend is constructed or required.

The async ``extension()`` factory runs on the event loop, so blocking
work (``mkdir``, SQLite open) is offloaded to a thread via
``asyncio.to_thread`` to avoid tripping dcode's blocking-call detector.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .cache import CodeCache
from .go_tracer import GoTracer
from .python_tracer import PythonTracer
from .tools import make_tools
from .tracer_base import Tracer

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI


def _venv_bin() -> str | None:
    """Return the bin directory of the running interpreter's venv, or None.

    uv-created venvs are not auto-activated, so ``ty`` is not on ``PATH``
    even though it lives next to ``sys.executable``.  The Python tracer
    uses this to locate the venv's ``ty`` binary directly.
    """
    exe = Path(sys.executable)
    if exe.parent.name in ("bin", "Scripts"):
        return str(exe.parent)
    return None


def _detect_language(root: str) -> str:
    """Detect the dominant language of the project."""
    root_path = Path(root)
    if (root_path / "go.mod").exists():
        return "go"
    # Default to Python if no go.mod found.
    return "python"


def _create_tracer(
    root: str,
    cache: CodeCache,
    language: str | None = None,
) -> Tracer:
    """Create the appropriate tracer for the detected language."""
    if language is None:
        language = _detect_language(root)

    if language == "go":
        return GoTracer(root=root, cache=cache)
    return PythonTracer(root=root, cache=cache, venv_bin=_venv_bin())


def get_code_tools(
    cwd: str,
    cache_path: str | None = None,
    language: str | None = None,
) -> list:
    """Build and return the six code-navigation tool functions.

    Used by the extension factory and by tests. Runs host-side: file
    access is direct on the host filesystem, and the language server runs
    as a local subprocess.
    """
    if cache_path is None:
        cache_path = str(Path(cwd) / ".metalgate" / "context_cache.db")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = CodeCache(cache_path)
    tracer = _create_tracer(root=cwd, cache=cache, language=language)
    return make_tools(tracer)


async def extension(api: ExtensionAPI) -> None:
    """Register the contextual symbol search tools for this session.

    Detects the project language (Go if ``go.mod`` is present, else
    Python), builds the language-specific tracer against ``api.cwd``, and
    registers the six code-navigation tools. Registers an ``on_shutdown``
    hook so the language-server subprocess is torn down when the session
    ends.
    """
    cache_path = str(Path(api.cwd) / ".metalgate" / "context_cache.db")
    # mkdir + SQLite open are blocking; offload them off the event loop.
    await asyncio.to_thread(
        lambda: Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    )
    cache = await asyncio.to_thread(CodeCache, cache_path)
    tracer = await asyncio.to_thread(_create_tracer, str(api.cwd), cache)

    for tool in make_tools(tracer):
        api.register_tool(tool)

    api.on_shutdown(tracer.stop)
