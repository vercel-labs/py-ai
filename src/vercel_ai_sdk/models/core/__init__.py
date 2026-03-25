"""Core model abstractions — LanguageModel, ImageModel, VideoModel."""

from . import media
from .image import ImageModel
from .llm import LanguageModel, StreamEvent, StreamHandler
from .media.base import MediaModel, MediaResult
from .model import Model
from .protocol import GenerateFn, Stream, StreamFn
from .registry import (
    get_generate_fn,
    get_stream_fn,
    register_generate,
    register_stream,
)
from .video import VideoModel

__all__ = [
    # Model data
    "Model",
    # Execution protocols
    "StreamFn",
    "GenerateFn",
    "Stream",
    # Registry
    "register_stream",
    "register_generate",
    "get_stream_fn",
    "get_generate_fn",
    # Legacy ABCs (still in use)
    "LanguageModel",
    "StreamEvent",
    "StreamHandler",
    "MediaModel",
    "MediaResult",
    "ImageModel",
    "VideoModel",
    "media",
]
