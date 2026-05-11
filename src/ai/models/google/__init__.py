"""Google Gemini provider.

Usage::

    from ai.models import google

    model = google("gemini-2.5-flash")
    ids = await google.list()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[google.tools.google_search()],
    ) as s:
        ...

The adapter module is loaded lazily to avoid pulling in the ``google-genai``
SDK at import time.
"""

from . import tools
from .provider import google

__all__ = ["google", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
