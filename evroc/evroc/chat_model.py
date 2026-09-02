"""evroc chat model — a thin ``ChatOpenAI`` subclass.

dcode's ``class_path`` mechanism imports this class via
``importlib.import_module("evroc")`` and instantiates it with
``ChatModel(model=<model_name>, **kwargs)`` where ``kwargs`` includes
``base_url`` and ``api_key`` resolved from the ``[models.providers.evroc]``
config table.

Because the evroc endpoint is OpenAI-compatible, we inherit everything from
``ChatOpenAI`` unchanged.
"""

from langchain_openai import ChatOpenAI


class ChatModel(ChatOpenAI):
    """OpenAI-compatible chat model pointed at the evroc inference endpoint.

    Instantiated by dcode's ``class_path`` provider mechanism. All constructor
    kwargs (``model``, ``base_url``, ``api_key``, ``temperature``, …) are
    forwarded by dcode from the ``[models.providers.evroc]`` config table and
    ``--model-params``.
    """
