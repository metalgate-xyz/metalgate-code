"""dcode extension factory for the dynamic tool loader.

Wires the dynamic tool scanner into the dcode extension registrar and exposes
agent-callable tools for writing and reloading tool files.

The async ``extension()`` factory runs on the event loop, so all filesystem
work (``mkdir``, ``stat``, ``rglob``, file reads) is offloaded to a thread via
``asyncio.to_thread`` to avoid tripping dcode's blocking-call detector.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .dynamic_tools import (
    _state_for,
    resolve_tool_directories,
    scan_and_register,
)

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI


async def extension(api: ExtensionAPI) -> None:
    """Register the dynamic tool loader and its agent-facing tools.

    Scans ``.metalgate/tools`` under the current project root and
    ``.metalgate/global_tools`` under the agent project root, importing every
    ``.py`` file and registering its top-level callables as model tools.
    """
    directories = resolve_tool_directories(api)
    # Ensure both scanned directories exist. The local tools dir is where
    # agent-authored tool files land; the agent repo's global_tools dir ships
    # repo-wide tools. Creating them up front also lets the sandbox bind them
    # read-write so the agent can edit global tool source when dcode runs from
    # a different project.
    # mkdir is a blocking call; run it off the event loop.
    await asyncio.to_thread(
        lambda: [d.mkdir(parents=True, exist_ok=True) for d in directories]
    )

    # scan_and_register does stat/rglob/import -- all blocking; offload it.
    await asyncio.to_thread(scan_and_register, api, directories)

    def reload_dynamic_tools() -> str:
        """Rescan the dynamic tools directories and register any new tool
        files, or files whose content changed. Call this after writing a new
        .py file with tool functions. Note: editing an already-loaded tool's
        code will NOT replace it live -- only brand-new function names are
        picked up without a /restart."""
        result = scan_and_register(api, directories)
        if result:
            return f"Registered/found: {', '.join(result)}"
        return "No new or changed tools found."

    def write_tool_file(filename: str, code: str) -> str:
        """Write Python source code to the current project's dynamic tools
        directory so it can be loaded as a tool file. `filename` should be a
        simple name like 'my_tool.py' (no path separators). Each top-level
        function in the file becomes a callable tool -- give it a clear name,
        type hints, and a docstring, since dcode infers the tool schema from
        the function signature and docstring. List names in __all__ if you
        only want a subset of top-level names exposed. After writing, call
        reload_dynamic_tools to activate it."""
        if "/" in filename or "\\" in filename or filename.startswith("."):
            return "Refusing: filename must be a plain name with no path separators."
        if not filename.endswith(".py"):
            filename += ".py"
        target = directories.local / filename
        target.write_text(code)
        return f"Wrote {target}. Call reload_dynamic_tools() to activate it."

    def list_dynamic_tools() -> str:
        """List the tool names currently registered from the dynamic tools
        directories."""
        names = _state_for(api).registered_names
        if not names:
            return "No dynamic tools registered yet."
        return "Registered dynamic tools: " + ", ".join(sorted(names))

    api.register_tool(reload_dynamic_tools)
    api.register_tool(write_tool_file)
    api.register_tool(list_dynamic_tools)
