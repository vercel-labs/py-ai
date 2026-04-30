"""AI Gateway provider.

Usage::

    from ai.models import ai_gateway

    model = ai_gateway("anthropic/claude-sonnet-4")
    ids = await ai_gateway.list()

The heavy ``.adapter`` module is loaded lazily so that ``import ai`` does
not pull in ``httpx`` and other I/O libraries at import time.  This matters
for sandboxed runtimes (e.g. Temporal workflow workers).
"""

from . import errors
from .provider import ai_gateway

__all__ = [
    "ai_gateway",
    "errors",
]


def __getattr__(name: str) -> object:
    if name == "generate":
        from .adapter import generate

        return generate
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
