"""Anthropic provider — adapter for the Anthropic messages API.

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

__all__: list[str] = []


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
