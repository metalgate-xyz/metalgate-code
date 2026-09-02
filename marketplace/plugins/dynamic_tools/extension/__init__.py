"""Dynamic tool-loading extension for dcode.

Registers a scanner that imports ``.py`` files from ``.metalgate/tools`` and
exposes their top-level callables as model tools, plus agent-callable tools to
write and reload tool files at runtime.
"""

from .factory import extension

__all__ = ["extension"]
