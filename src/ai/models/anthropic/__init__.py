"""Anthropic provider.

Usage::

    from ai.models import anthropic

    model = anthropic("claude-sonnet-4-6")
    ids = await anthropic.list()

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from __future__ import annotations

from .provider import anthropic

__all__ = ["anthropic"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
