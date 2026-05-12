"""OpenAI provider.

Usage::

    from ai.providers import openai, openai_like

    model = openai("gpt-5.4")
    model = openai_like(name="local", base_url="http://localhost:11434/v1")("llama3")
    ids = await openai.list()

The adapter module is loaded lazily to avoid pulling in the ``openai``
SDK at import time.
"""

from . import tools
from .provider import OpenAICompatibleProvider, openai, openai_like

__all__ = ["OpenAICompatibleProvider", "openai", "openai_like", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
