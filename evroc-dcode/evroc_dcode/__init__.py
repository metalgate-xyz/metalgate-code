"""evroc-dcode model provider for deepagents code.

Exposes ``ChatModel`` — a ``ChatOpenAI`` subclass that routes requests to the
evroc inference endpoint — and ``fetch_models`` for dynamic model discovery.

Used with ``config.toml``::

    [models.providers.evroc]
    class_path = "evroc_dcode:ChatModel"
    api_key_env = "EVROC_API_KEY"
    base_url = "https://models.think.cloud.evroc.com/v1"
"""

from .chat_model import ChatModel
from .models import fetch_models, load_model_ids, save_models

__all__ = ["ChatModel", "fetch_models", "load_model_ids", "save_models"]
