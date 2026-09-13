"""pytest config: make the extension importable without polluting the marketplace.

The plugin lives under ``marketplace/plugins/context_tools/extension/``. The
marketplace directory is a plugin catalog, not a Python package, so we do not
add ``__init__.py`` files there. Instead we load the extension package directly
by file path (the same approach dcode's own ``loader.py`` uses) and register it
in ``sys.modules`` under the dotted path the tests import.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_TESTS = Path(__file__).resolve().parent  # .../context_tools/tests
_PLUGIN = _TESTS.parent  # .../context_tools
_EXT = _PLUGIN / "extension"  # .../context_tools/extension
_PLUGINS = _PLUGIN.parent  # .../plugins
_MARKETPLACE = _PLUGINS.parent  # .../marketplace

# Stub the intermediate namespace packages so `from marketplace.plugins...`
# resolves. Each gets an __path__ pointing at its directory.
for _name, _path in (
    ("marketplace", _MARKETPLACE),
    ("marketplace.plugins", _PLUGINS),
    ("marketplace.plugins.context_tools", _PLUGIN),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = [str(_path)]
        sys.modules[_name] = _mod

# Load the real extension package from its __init__.py.
_PKG = "marketplace.plugins.context_tools.extension"
if _PKG not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _PKG,
        _EXT / "__init__.py",
        submodule_search_locations=[str(_EXT)],
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"could not load extension package from {_EXT}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_PKG] = _module
    _spec.loader.exec_module(_module)
