"""Model profiles for the evroc-dcode provider.

Reads ``.evroc/models.json`` at import time and builds ``_PROFILES``
dynamically.  ``run.sh`` fetches models from the evroc API and writes
the JSON file before launching dcode.
"""

import json
from pathlib import Path
from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {}

_models_file = Path(".evroc") / "models.json"
if _models_file.exists():
    for _id in json.loads(_models_file.read_text(encoding="utf-8")):
        _PROFILES[_id] = {
            "name": _id,
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        }
