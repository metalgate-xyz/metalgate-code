"""Model profiles for the evroc-dcode provider.

Reads the models file from ``EVROC_DATA_DIR`` at import time and builds
``_PROFILES`` dynamically.  ``run.sh`` fetches models from the evroc API,
writes ``models.json`` into the data dir, and exports ``EVROC_DATA_DIR``
before launching dcode.

The path must be independent of the process cwd: to decouple the lookup
from the dcode discovering process.
"""

import json
import os
from pathlib import Path
from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {}

_data_dir = Path(os.environ.get("EVROC_DATA_DIR", ".evroc"))
_models_file = _data_dir / "models.json"
if _models_file.is_file():
    for _id in json.loads(_models_file.read_text(encoding="utf-8")):
        _PROFILES[_id] = {
            "name": _id,
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        }
