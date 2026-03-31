"""models2 — composable model layer.

Usage::

    from vercel_ai_sdk import models2 as m
    from vercel_ai_sdk.types import Message, TextPart

    model = m.Model(
        id="anthropic/claude-sonnet-4",
        api="ai-gateway",
        provider="ai-gateway",
    )
    msgs = [Message(role="user", parts=[TextPart(text="hello")])]

    # stream — auto-creates client from env vars
    async for msg in m.stream(model, msgs):
        print(msg.text_delta, end="")

    # buffer the whole response
    result = await m.buffer(m.stream(model, msgs))
    print(result.text)

    # explicit client
    client = m.Client(base_url="https://custom.example.com/v3/ai", api_key="sk-...")
    async for msg in m.stream(model, msgs, client=client):
        ...
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

from ..types import messages as messages_
from ..types import tools as tools_
from .core.client import Client
from .core.model import Model, ModelCost
from .core.wire import GenerateFn, StreamFn

# ---------------------------------------------------------------------------
# Wire registry — maps api string → wire function.
# Wire modules are imported lazily on first use.
# ---------------------------------------------------------------------------

_stream_wires: dict[str, StreamFn] = {}
_generate_wires: dict[str, GenerateFn] = {}
_wires_loaded = False


def _ensure_wires() -> None:
    """Lazily register built-in wire functions on first call."""
    global _wires_loaded  # noqa: PLW0603
    if _wires_loaded:
        return
    _wires_loaded = True

    from .wires import ai_gateway_v3

    _stream_wires["ai-gateway"] = ai_gateway_v3.stream
    _generate_wires["ai-gateway"] = ai_gateway_v3.generate


# ---------------------------------------------------------------------------
# Provider defaults — base URLs and env var names for auto-client creation.
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "ai-gateway": ("https://ai-gateway.vercel.sh/v3/ai", "AI_GATEWAY_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _auto_client(model: Model) -> Client:
    """Create a :class:`Client` from env vars for the given model's provider."""
    defaults = _PROVIDER_DEFAULTS.get(model.provider)
    if defaults is None:
        raise ValueError(
            f"No default client config for provider {model.provider!r}. "
            f"Pass an explicit client= argument."
        )
    base_url, env_var = defaults
    return Client(base_url=base_url, api_key=os.environ.get(env_var))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream(
    model: Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> AsyncGenerator[messages_.Message]:
    """Stream an LLM response.

    Resolves the wire function from ``model.api``, auto-creates a
    :class:`Client` from env vars if none is provided, and yields
    ``Message`` snapshots.
    """
    _ensure_wires()
    c = client or _auto_client(model)
    wire_fn = _stream_wires.get(model.api)
    if wire_fn is None:
        registered = ", ".join(sorted(_stream_wires)) or "(none)"
        raise KeyError(
            f"No stream wire registered for api={model.api!r}. Registered: {registered}"
        )
    async for msg in wire_fn(
        c, model, messages, tools=tools, output_type=output_type, **kwargs
    ):
        yield msg


async def generate(
    model: Model,
    messages: list[messages_.Message],
    *,
    client: Client | None = None,
    **kwargs: Any,
) -> messages_.Message:
    """Generate a response (images, video, etc.).

    Resolves the wire function from ``model.api``, auto-creates a
    :class:`Client` from env vars if none is provided.
    """
    _ensure_wires()
    c = client or _auto_client(model)
    wire_fn = _generate_wires.get(model.api)
    if wire_fn is None:
        registered = ", ".join(sorted(_generate_wires)) or "(none)"
        raise KeyError(
            f"No generate wire registered for api={model.api!r}. "
            f"Registered: {registered}"
        )
    return await wire_fn(c, model, messages, **kwargs)


async def buffer(gen: AsyncGenerator[messages_.Message]) -> messages_.Message:
    """Drain a stream and return the final ``Message``.

    Raises :class:`ValueError` if the stream yields nothing.
    """
    result: messages_.Message | None = None
    async for msg in gen:
        result = msg
    if result is None:
        raise ValueError("empty stream")
    return result


__all__ = [
    # Core types
    "Client",
    "GenerateFn",
    "Model",
    "ModelCost",
    "StreamFn",
    # Public API
    "buffer",
    "generate",
    "stream",
]
