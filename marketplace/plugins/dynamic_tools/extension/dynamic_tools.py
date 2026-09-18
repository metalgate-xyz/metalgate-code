"""Dynamic tool-loading core for dcode.

Scans two directories for ``.py`` files, imports each one, and registers every
top-level callable (or everything listed in ``__all__``) as a model tool via
``api.register_tool()``.

Two directories are scanned, in order, de-duplicated by path equality:

1. **Local project tools** -- ``<current project>/.metalgate/tools``. This is
   the project the agent is currently editing. Agent-authored tool files
   (written by ``write_tool_file``) land here.
2. **Global agent tools** -- ``<agent repo>/.metalgate/global_tools``. The
   agent project root is derived from the parent of ``DEEPAGENTS_HOME``
   (``run.sh`` sets ``DEEPAGENTS_HOME=<repo>/.metalgate``, so the parent is the
   repo root). Tools defined here ship with the agent repo and are available in
   every session, regardless of the current project. Falls back to the current
   project root when ``DEEPAGENTS_HOME`` is unset.

Splitting the two on different directory names (``tools`` vs ``global_tools``)
keeps the agent repo's own tools global without sweeping every
``.metalgate/tools`` directory the local project happens to use into the global
namespace.

IMPORTANT CAVEATS (read before relying on this):

1. ``register_tool()`` calls made AFTER initial extension setup still take
   effect live -- tools registered after startup appear on the next model
   request. That is what makes ``reload_dynamic_tools()`` work without
   ``/restart``.

2. Between extensions/registrations, the first registration for a tool name
   wins. That means if the agent edits an EXISTING tool's code and calls
   ``reload_dynamic_tools()``, the old version stays active -- only brand-new
   function names get picked up live. To actually replace a tool's
   implementation you need a full ``/restart``. This module tracks file mtimes
   so it re-imports changed files, but it silently skips re-registering any
   name that is already registered.

3. This extension runs arbitrary Python with your user account's permissions,
   and extension tools are NOT automatically added to the human-approval map.
   If the agent is writing its own tool code and that code does anything
   sensitive (network, filesystem outside the sandbox, subprocess, etc.),
   there is no approval gate unless you add one yourself (e.g. via
   ``register_middleware``). Treat agent-authored tool code as untrusted until
   you have reviewed it.

4. Requires ``DEEPAGENTS_CODE_EXPERIMENTAL=1`` and extension discovery enabled.
   Packaged as a dcode marketplace plugin; ``run.sh`` registers the marketplace
   and installs the plugin::

        dcode plugin marketplace add ./marketplace
        dcode plugin install dynamic_tools@evroc-extensions
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI

# Directory watched under the CURRENT project root (the project being edited).
# Agent-authored tool files (written by ``write_tool_file``) land here.
LOCAL_TOOLS_DIRNAME = ".metalgate/tools"

# Directory watched under the AGENT project root (the repo that ships this
# plugin). Tools here are global -- available in every session regardless of
# the current project. Using a different name from LOCAL_TOOLS_DIRNAME keeps the
# agent repo's own tools global without pulling every local project's
# ``.metalgate/tools`` into the global namespace.
GLOBAL_TOOLS_DIRNAME = ".metalgate/global_tools"


class ToolDirectories(NamedTuple):
    """The two directories the scanner watches, by role.

    ``local`` is the current project's tools dir; ``global`` is the agent
    repo's global tools dir. They differ in the final path segment, so both
    are always distinct.
    """

    local: Path
    global_: Path


# Per-registry dedup state. dcode (>=0.1.66) builds a fresh ExtensionRegistry
# per workspace binding, invoking this extension factory more than once in the
# same process. Dedup state must therefore be scoped to the registry instance,
# not held in a process-global set: a brand-new registry that shares nothing
# with the previous (now-discarded) one would otherwise skip every tool file
# because a process-global mtime/name set already recorded them -- silently
# dropping dynamically-loaded tools (e.g. ``.metalgate/tools/*.py``) from the
# registry that actually serves the request.
#
# The state is stored directly on the ``ExtensionRegistry`` object under
# ``_dynamic_tools_state`` (a regular class with ``__dict__``), so it lives and
# dies with that registry -- no ``id()`` reuse hazard, no stale entries to prune.
# A fresh registry starts with no state, so its first ``scan_and_register``
# registers every tool file into it even if an earlier registry in this process
# already saw the same files.
#
# ``_loaded_mtimes`` and ``_registered_names`` below are retained only for
# backward compatibility with callers that clear them between runs; the live
# dedup state lives on the registry and these module globals are unused.
_loaded_mtimes: dict[str, float] = {}
_registered_names: set[str] = set()

_REGISTRY_STATE_ATTR = "_dynamic_tools_state"


class _RegistryState:
    """Dedup state scoped to one ``ExtensionRegistry``.

    ``loaded_mtimes`` skips re-importing an unchanged file on rescan (so
    ``reload_dynamic_tools()`` only re-imports files whose mtime changed).
    ``registered_names`` enforces first-registration-wins *within this
    registry*: editing an existing tool's code and reloading keeps the original
    implementation live (a full ``/restart`` is required to swap it).
    """

    __slots__ = ("loaded_mtimes", "registered_names")

    def __init__(self) -> None:
        self.loaded_mtimes: dict[str, float] = {}
        self.registered_names: set[str] = set()


def _state_for(api: ExtensionAPI) -> _RegistryState:
    """Return the dedup state bound to ``api``'s registry, creating it if new.

    A fresh registry (a new workspace runtime) gets a fresh state, so tool files
    register into it even when an earlier registry in this process already saw
    them.
    """
    registry = api._registry  # type: ignore[attr-defined]
    state = getattr(registry, _REGISTRY_STATE_ATTR, None)
    if state is None:
        state = _RegistryState()
        setattr(registry, _REGISTRY_STATE_ATTR, state)
    return state


def _agent_project_root(api: ExtensionAPI) -> Path:
    """Return the agent project root (the repo that ships this plugin).

    Derived from the parent of ``DEEPAGENTS_HOME``: ``run.sh`` sets
    ``DEEPAGENTS_HOME=<repo>/.metalgate``, so the parent is the repo root.
    Falls back to the current project root (``api.cwd``) when
    ``DEEPAGENTS_HOME`` is unset.
    """
    home = os.environ.get("DEEPAGENTS_HOME")
    if home:
        root = Path(home).parent
        if root.is_dir():
            return root
    return Path(api.cwd)


def resolve_tool_directories(api: ExtensionAPI) -> ToolDirectories:
    """Return the tool directories to scan, by role.

    ``local`` is the current project's ``.metalgate/tools`` dir; ``global`` is
    the agent repo's ``.metalgate/global_tools`` dir.
    """
    local = Path(api.cwd) / LOCAL_TOOLS_DIRNAME
    global_ = _agent_project_root(api) / GLOBAL_TOOLS_DIRNAME
    return ToolDirectories(local=local, global_=global_)


def _discover_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.py") if p.is_file())


def _import_module(path: Path):
    # unique module name per file so reimports don't collide
    mod_name = f"dcode_dynamic_tools.{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _extract_callables(module):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    found = []
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            found.append((name, obj))
    return found


def scan_and_register(api: ExtensionAPI, directories: ToolDirectories) -> list[str]:
    """Scan ``directories`` and register any new/changed tools.

    Returns a list of newly registered tool names (plus a trailing
    ``[errors: ...]`` entry if any file failed to import).
    """
    newly_registered: list[str] = []
    errors: list[str] = []
    state = _state_for(api)

    for directory in directories:
        for path in _discover_files(directory):
            key = str(path)
            mtime = path.stat().st_mtime
            if state.loaded_mtimes.get(key) == mtime:
                continue  # unchanged since last scan of THIS registry

            try:
                module = _import_module(path)
            except Exception as exc:  # noqa: BLE001 - surface to caller, don't crash setup
                errors.append(f"{path.name}: {exc}")
                continue

            for name, fn in _extract_callables(module):
                tool_name = getattr(fn, "name", None) or getattr(fn, "__name__", name)
                if tool_name in state.registered_names:
                    # first registration wins within this registry -- can't
                    # hot-swap; needs /restart
                    continue
                api.register_tool(fn)
                state.registered_names.add(tool_name)
                newly_registered.append(tool_name)

            state.loaded_mtimes[key] = mtime

    if errors:
        newly_registered.append(f"[errors: {'; '.join(errors)}]")

    return newly_registered
