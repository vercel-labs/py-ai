"""AI Gateway provider — adapter for the Vercel AI Gateway v3 protocol."""

from .generate import GenerateParams, ImageParams, VideoParams, generate
from .stream import stream

__all__ = [
    "GenerateParams",
    "ImageParams",
    "VideoParams",
    "generate",
    "stream",
]
