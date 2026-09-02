"""Fetch evroc models and write .evroc/models.json for the dcode /model switcher.

Run by ``run.sh`` before launching dcode. A failed fetch logs a warning and
exits 0 so dcode can still launch against a stale ``models.json`` rather than
aborting the whole session.
"""

from __future__ import annotations

import logging

from evroc import save_models

logging.basicConfig(level=logging.INFO, format="%(message)s")

ids = save_models()
print(f"Fetched {len(ids)} models")
