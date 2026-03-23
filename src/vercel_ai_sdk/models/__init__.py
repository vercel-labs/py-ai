"""Model adapters — standalone LLM streaming layer.

Provides the LanguageModel ABC and concrete provider adapters.
Depends only on types/, never on agents/.
"""

from . import ai_gateway, anthropic, core, openai
from .core import (
    ImageModel,
    LanguageModel,
    MediaModel,
    MediaResult,
    StreamEvent,
    StreamHandler,
    VideoModel,
)

__all__ = [
    # Core abstractions
    "LanguageModel",
    "StreamEvent",
    "StreamHandler",
    "MediaModel",
    "MediaResult",
    "ImageModel",
    "VideoModel",
    "core",
    # Provider adapters
    "openai",
    "anthropic",
    "ai_gateway",
]
