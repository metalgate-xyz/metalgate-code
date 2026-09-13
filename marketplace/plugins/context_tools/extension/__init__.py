"""Real-time contextual symbol search tools for dcode.

Exposes six code-navigation tools — goto_definition, get_file_outline,
get_source, get_callers, get_callees, find_symbol — backed by the ty
(Python) and gopls (Go) language servers plus tree-sitter.

The dcode extension factory (:func:`extension`) registers the tools for
the current session's project root. The language servers run on the host
as local subprocesses; file access is direct on the host filesystem.
"""

from .factory import extension, get_code_tools
from .go_tracer import GoTracer
from .python_tracer import PythonTracer
from .tracer_base import Tracer

__all__ = ["GoTracer", "PythonTracer", "Tracer", "extension", "get_code_tools"]
