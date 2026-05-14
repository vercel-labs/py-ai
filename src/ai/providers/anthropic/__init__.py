"""Anthropic provider.

Usage::

    import ai
    from ai.providers.anthropic import tools as anthropic_tools

    model = ai.get_model("anthropic:claude-sonnet-4-6")
    provider = ai.get_provider("anthropic", base_url="https://anthropic.example.com")
    model = ai.Model("claude-sonnet-4-6", provider=provider)
    ids = await ai.get_provider("anthropic").list_models()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[anthropic_tools.web_search(max_uses=5)],
    ) as s:
        ...

The protocol module is loaded lazily by provider methods.
"""

from . import tools
from .provider import AnthropicCompatibleProvider

__all__ = ["AnthropicCompatibleProvider", "tools"]
