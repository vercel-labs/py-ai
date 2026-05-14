"""OpenAI provider.

Usage::

    import ai

    model = ai.get_model("openai:gpt-5.4")
    provider = ai.get_provider("openai", base_url="http://localhost:11434/v1")
    model = ai.Model("llama3", provider=provider)
    ids = await ai.get_provider("openai").list_models()

The protocol module is loaded lazily by provider methods.
"""

from . import tools
from .provider import OpenAICompatibleProvider

__all__ = ["OpenAICompatibleProvider", "tools"]
