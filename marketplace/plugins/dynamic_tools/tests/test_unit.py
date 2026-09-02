"""Unit tests for the dynamic_tools extension.

Drives the real ``ExtensionAPI`` + ``ExtensionRegistry`` directly (no dcode
process, no model), against temp directories that stand in for the two project
roots the scanner watches.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from deepagents_code.extensions.api import ExtensionAPI, ExtensionMode
from deepagents_code.extensions.discovery import SourceInfo
from deepagents_code.extensions.loader import load_extension
from deepagents_code.extensions.registry import (
    ExtensionRegistry,
    SourceScope,
)

from marketplace.plugins.dynamic_tools.extension import dynamic_tools as dt


# Importing the package triggers the extension factory; reset shared state
# between tests so registrations don't leak.
@pytest.fixture(autouse=True)
def _reset_state():
    dt._loaded_mtimes.clear()
    dt._registered_names.clear()
    yield
    dt._loaded_mtimes.clear()
    dt._registered_names.clear()


def _make_api(cwd: Path, tmp_path: Path) -> tuple[ExtensionAPI, ExtensionRegistry]:
    """Build a real ExtensionAPI backed by a fresh registry.

    ``cwd`` is what the extension sees as the current project root.
    ``tmp_path`` is just used to derive a throwaway source path.
    """
    registry = ExtensionRegistry()
    source = SourceInfo(
        path=tmp_path / "extension.py",
        is_package=False,
        source_id="dynamic_tools@test",
        scope=SourceScope.TEMPORARY,
        version="0.1.0",
    )
    api = ExtensionAPI(
        registry,
        source,
        cwd=cwd,
        mode=ExtensionMode.HEADLESS,
    )
    return api, registry


async def _run_extension(api: ExtensionAPI) -> ExtensionRegistry:
    """Invoke the package's real ``extension`` factory against ``api``."""
    from marketplace.plugins.dynamic_tools.extension import extension

    await extension(api)
    return api._registry  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tool file discovery + registration
# ---------------------------------------------------------------------------

HELLO_TOOL = textwrap.dedent(
    """
    def say_hello(name: str) -> str:
        '''Greet someone.'''
        return f"hello {name}"
    """
)


async def test_loads_tool_from_current_project_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    (tmp_path / ".metalgate" / "tools").mkdir(parents=True)
    (tmp_path / ".metalgate" / "tools" / "hello.py").write_text(HELLO_TOOL)

    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)

    unit = registry.find_tool("say_hello")
    assert unit is not None, registry.tool_units()
    # The registered BaseTool is invocable directly.
    assert unit.unit.invoke({"name": "marx"}) == "hello marx"


async def test_loads_tools_from_both_roots(tmp_path, monkeypatch):
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    monkeypatch.setenv("DEEPAGENTS_HOME", str(agent_root / ".metalgate"))

    # Current project contributes one tool under .metalgate/tools; the agent
    # repo contributes another under .metalgate/global_tools.
    current = tmp_path / "current"
    (current / ".metalgate" / "tools").mkdir(parents=True)
    (current / ".metalgate" / "tools" / "cur.py").write_text(
        "def cur_tool() -> str:\n    '''from current root'''\n    return 'cur'"
    )
    (agent_root / ".metalgate" / "global_tools").mkdir(parents=True)
    (agent_root / ".metalgate" / "global_tools" / "agent.py").write_text(
        "def agent_tool() -> str:\n    '''from agent root'''\n    return 'agent'"
    )

    api, registry = _make_api(current, tmp_path)
    await _run_extension(api)

    names = {u.name for u in registry.tool_units()}
    assert {"cur_tool", "agent_tool"} <= names


async def test_local_tools_dir_is_not_swept_into_global(tmp_path, monkeypatch):
    """A local .metalgate/tools dir must NOT be loaded as global tools when the
    agent root differs from the current root."""
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    monkeypatch.setenv("DEEPAGENTS_HOME", str(agent_root / ".metalgate"))

    current = tmp_path / "current"
    (current / ".metalgate" / "tools").mkdir(parents=True)
    (current / ".metalgate" / "tools" / "local_only.py").write_text(
        "def local_only_tool() -> str:\n    '''local only'''\n    return 'local'"
    )
    # The agent root has NO .metalgate/global_tools -- so global_tools is absent.
    # The local project's .metalgate/tools must still load (it is the local dir).
    api, registry = _make_api(current, tmp_path)
    await _run_extension(api)

    assert registry.find_tool("local_only_tool") is not None
    # And the local tools dir must NOT be scanned a second time as global_tools.
    assert len([u for u in registry.tool_units() if u.name == "local_only_tool"]) == 1


async def test_dedups_when_roots_collapse(tmp_path, monkeypatch):
    # DEEPAGENTS_HOME under cwd -> agent root == current root. With distinct
    # dir names (tools vs global_tools) both are scanned, but each file is only
    # imported once (different dirs, no overlap in this case).
    monkeypatch.setenv("DEEPAGENTS_HOME", str(tmp_path / ".metalgate"))
    (tmp_path / ".metalgate" / "tools").mkdir(parents=True)
    (tmp_path / ".metalgate" / "tools" / "hello.py").write_text(HELLO_TOOL)

    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)

    assert registry.find_tool("say_hello") is not None
    # No duplicate registration / no crash from double-import.
    assert len([u for u in registry.tool_units() if u.name == "say_hello"]) == 1


# ---------------------------------------------------------------------------
# __all__ gating + first-registration-wins
# ---------------------------------------------------------------------------

GATED_MODULE = textwrap.dedent(
    """
    __all__ = ["exported"]

    def exported() -> str:
        '''should be registered'''
        return "out"

    def hidden() -> str:
        '''should NOT be registered'''
        return "secret"
    """
)


async def test_all_gates_unlisted_callables(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    (tmp_path / ".metalgate" / "tools").mkdir(parents=True)
    (tmp_path / ".metalgate" / "tools" / "gated.py").write_text(GATED_MODULE)

    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)

    assert registry.find_tool("exported") is not None
    assert registry.find_tool("hidden") is None


async def test_first_registration_wins_on_reload(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    tools_dir = tmp_path / ".metalgate" / "tools"
    tools_dir.mkdir(parents=True)
    tools_dir.joinpath("hello.py").write_text(HELLO_TOOL)

    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)
    original = registry.find_tool("say_hello").unit

    # Edit the file in place and call reload_dynamic_tools(). The mtime changes
    # so the file is re-imported, but the tool NAME is already registered, so
    # the old implementation must stay (first registration wins).
    tools_dir.joinpath("hello.py").write_text(
        textwrap.dedent(
            """
            def say_hello(name: str) -> str:
                '''NEW implementation'''
                return f"hi {name}"
            """
        )
    )
    reload_tool = registry.find_tool("reload_dynamic_tools").unit
    result = reload_tool.invoke({})
    assert "No new or changed tools found" in result or "Registered" not in result
    assert registry.find_tool("say_hello").unit is original


async def test_reload_picks_up_brand_new_tool(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    tools_dir = tmp_path / ".metalgate" / "tools"
    tools_dir.mkdir(parents=True)
    tools_dir.joinpath("hello.py").write_text(HELLO_TOOL)

    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)
    assert registry.find_tool("say_hello") is not None

    # Write a NEW file with a NEW function name.
    tools_dir.joinpath("bye.py").write_text(
        "def say_bye() -> str:\n    '''bye'''\n    return 'bye'"
    )
    reload_tool = registry.find_tool("reload_dynamic_tools").unit
    out = reload_tool.invoke({})
    assert "say_bye" in out
    assert registry.find_tool("say_bye") is not None


# ---------------------------------------------------------------------------
# write_tool_file + list_dynamic_tools
# ---------------------------------------------------------------------------


async def test_write_tool_file_rejects_path_separators(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)

    write_tool = registry.find_tool("write_tool_file").unit
    for bad in ["../x.py", "a/b.py", ".hidden.py"]:
        assert "Refusing" in write_tool.invoke({"filename": bad, "code": "x = 1"})


async def test_write_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    api, registry = _make_api(tmp_path, tmp_path)
    await _run_extension(api)

    write_tool = registry.find_tool("write_tool_file").unit
    out = write_tool.invoke(
        {
            "filename": "dyn.py",
            "code": "def dyn_tool() -> str:\n    '''runtime-added'''\n    return 'dyn'",
        }
    )
    assert "Wrote" in out

    reload_tool = registry.find_tool("reload_dynamic_tools").unit
    assert "dyn_tool" in reload_tool.invoke({})
    assert registry.find_tool("dyn_tool") is not None

    list_tool = registry.find_tool("list_dynamic_tools").unit
    listing = list_tool.invoke({})
    assert "dyn_tool" in listing


async def test_import_errors_are_surfaced_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    tools_dir = tmp_path / ".metalgate" / "tools"
    tools_dir.mkdir(parents=True)
    tools_dir.joinpath("broken.py").write_text("raise RuntimeError('boom')")
    tools_dir.joinpath("good.py").write_text(
        "def good_tool() -> str:\n    '''works'''\n    return 'ok'"
    )

    api, registry = _make_api(tmp_path, tmp_path)
    # Must not raise despite broken.py.
    await _run_extension(api)

    assert registry.find_tool("good_tool") is not None
    assert registry.find_tool("broken_module") is None  # no name registered from it


# ---------------------------------------------------------------------------
# Full loader path (ExtensionAPI construction by dcode's own loader)
# ---------------------------------------------------------------------------


async def test_loads_via_dcode_loader(tmp_path, monkeypatch):
    """Exercise the real load_extension() path, not a hand-built API."""
    monkeypatch.delenv("DEEPAGENTS_HOME", raising=False)
    (tmp_path / ".metalgate" / "tools").mkdir(parents=True)
    (tmp_path / ".metalgate" / "tools" / "hello.py").write_text(HELLO_TOOL)

    plugin_root = (
        Path(__file__).resolve().parents[1]  # .../marketplace/plugins/dynamic_tools
    )
    entry = plugin_root / "extension" / "__init__.py"
    source = SourceInfo(
        path=entry,
        is_package=True,
        source_id="dynamic_tools@evroc-extensions",
        scope=SourceScope.TEMPORARY,
        version="0.1.0",
    )
    registry = ExtensionRegistry()
    api = await load_extension(
        source, registry, cwd=tmp_path, mode=ExtensionMode.HEADLESS
    )
    assert registry.find_tool("say_hello") is not None
    assert registry.find_tool("reload_dynamic_tools") is not None
    api._deactivate()
