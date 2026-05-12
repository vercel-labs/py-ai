"""Anthropic provider.

Usage::

    from ai.providers import anthropic, anthropic_like

    model = anthropic("claude-sonnet-4-6")
    model = anthropic_like(name="custom", base_url="https://anthropic.example.com")(
        "claude-sonnet-4-6"
    )
    ids = await anthropic.list()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[anthropic.tools.web_search(max_uses=5)],
    ) as s:
        ...

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from . import tools
from .provider import AnthropicCompatibleProvider, anthropic, anthropic_like

__all__ = ["AnthropicCompatibleProvider", "anthropic", "anthropic_like", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
