"""Model adapters — standalone LLM streaming layer.

Provides the LanguageModel ABC and concrete provider adapters.
Depends only on types/, never on agents/.

Module-level API
~~~~~~~~~~~~~~~~

.. code-block:: python

    import vercel_ai_sdk as ai

    model = ai.models.Model(id="gpt-4o", api="openai", provider="openai")
    s = ai.models.stream(model, messages)
    async for msg in s:
        ...

    result = await ai.models.generate(model, messages, n=2)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pydantic

from ..types import messages as messages_
from ..types import tools as tools_
from . import ai_gateway, anthropic, core, openai
from .core import (
    GenerateFn,
    ImageModel,
    LanguageModel,
    MediaModel,
    MediaResult,
    Model,
    Stream,
    StreamEvent,
    StreamFn,
    StreamHandler,
    VideoModel,
    get_generate_fn,
    get_stream_fn,
    register_generate,
    register_stream,
)

# ── Module-level dispatch functions ───────────────────────────────


def stream(
    model: Model,
    messages: list[messages_.Message],
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
) -> Stream:
    """Stream an LLM response for the given model.

    Looks up the registered :class:`StreamFn` for ``model.api`` and
    returns a :class:`Stream` that can be async-iterated *or* awaited.
    """
    fn = get_stream_fn(model.api)
    return Stream(fn(model, messages, tools=tools, output_type=output_type))


async def generate(
    model: Model,
    messages: list[messages_.Message],
    **kwargs: Any,
) -> messages_.Message:
    """Generate a response (image, video, etc.) for the given model.

    Looks up the registered :class:`GenerateFn` for ``model.api`` and
    returns the resulting :class:`Message`.
    """
    fn = get_generate_fn(model.api)
    return await fn(model, messages, **kwargs)


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
    # Dispatch
    "stream",
    "generate",
    # Legacy ABCs (still in use)
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
