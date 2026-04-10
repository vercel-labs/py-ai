"""AI Gateway provider — adapter for the Vercel AI Gateway v3 protocol.

Heavy adapter modules (``.generate``, ``.stream``) are loaded lazily so that
``import ai`` does not pull in ``httpx`` and other I/O libraries at import
time.  This matters for sandboxed runtimes (e.g. Temporal workflow workers).
"""

from . import errors
from .types import GenerateParams, ImageParams, VideoParams

__all__ = [
    "GenerateParams",
    "ImageParams",
    "VideoParams",
    "errors",
]


def __getattr__(name: str) -> object:
    if name == "generate":
        from .generate import generate

        return generate
    if name == "stream":
        from .stream import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
