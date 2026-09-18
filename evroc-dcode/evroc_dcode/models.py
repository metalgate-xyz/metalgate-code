"""Dynamic model discovery for the evroc API.

The evroc-dcode package fetches model IDs from the evroc inference endpoint
and persists them to ``.evroc/models.json`` for dcode's ``/model`` switcher.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://models.think.cloud.evroc.com/v1"

EVROC_DATA_DIR = Path(os.environ.get("EVROC_DATA_DIR", ".evroc"))
MODELS_FILE = EVROC_DATA_DIR / "models.json"


def fetch_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30,
) -> list[str]:
    """Fetch available model IDs from the evroc API.

    Returns an empty list on failure (missing key, network error); the cause
    is logged at WARNING level so a stale ``models.json`` launch is
    distinguishable from success.
    """
    api_key = api_key or os.environ.get("EVROC_API_KEY", "")
    if not api_key:
        logger.warning("EVROC_API_KEY is not set; cannot fetch models")
        return []

    base_url = base_url or os.environ.get("EVROC_BASE_URL", DEFAULT_BASE_URL)
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch evroc models: %s", e)
        return []

    print(response.json())

    return [m["id"] for m in response.json().get("data", [])]


def save_models(models: list[str] | None = None) -> list[str]:
    """Fetch (if needed) and persist model IDs to ``.evroc/models.json``.

    Returns the IDs written; empty list if nothing was saved.
    """
    if models is None:
        models = fetch_models()
    if not models:
        return []

    EVROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_FILE.write_text(json.dumps(models), encoding="utf-8")
    logger.info("Saved %d evroc models to %s", len(models), MODELS_FILE)
    return models


def load_model_ids() -> list[str]:
    """Load model IDs from ``.evroc/models.json`` (empty list if absent)."""
    if not MODELS_FILE.exists():
        return []
    return json.loads(MODELS_FILE.read_text(encoding="utf-8"))
