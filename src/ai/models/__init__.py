"""models — composable model layer.

Usage::

    import ai
    from ai.types import Message, TextPart

    # look up a model from the catalog
    opus = ai.model("ai-gateway", "anthropic/claude-opus-4-6")

    msgs = [Message(role="user", parts=[TextPart(text="hello")])]

    # stream — auto-creates client from env vars
    s = await ai.stream(opus, msgs)
    async for msg in s:
        print(msg.text_delta, end="")

    # buffer the whole response
    result = await ai.models.buffer(await ai.stream(opus, msgs))
    print(result.text)

    # explicit client
    client = ai.Client(
        base_url="https://custom.example.com/v3/ai", api_key="sk-...",
    )
    s = await ai.stream(opus, msgs, client=client)
    async for msg in s:
        ...
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

from .. import middleware as middleware_
from ..types import messages as messages_
from ..types import tools as tools_
from ..types.stream import StreamResultLike
from .ai_gateway.types import GenerateParams, ImageParams, VideoParams
from .core.catalog import get_models, get_providers, register_catalog
from .core.catalog import model as model
from .core.client import Client
from .core.model import Model, ModelCost
from .core.proto import CheckConnFn, GenerateFn, StreamFn

# ---------------------------------------------------------------------------
# Adapter registry — maps adapter string → adapter function.
# Adapter modules are imported lazily on first use.
# ---------------------------------------------------------------------------

_stream_adapters: dict[str, StreamFn] = {}
_generate_adapters: dict[str, GenerateFn] = {}
_adapters_loaded = False


def _ensure_adapters() -> None:
    """Lazily register built-in adapter functions on first call."""
    global _adapters_loaded  # noqa: PLW0603
    if _adapters_loaded:
        return
    _adapters_loaded = True

    from .ai_gateway.generate import generate as ai_gw_generate
    from .ai_gateway.stream import stream as ai_gw_stream
    from .anthropic.adapter import stream as anthropic_stream
    from .openai.adapter import stream as openai_stream

    _stream_adapters["ai-gateway-v3"] = ai_gw_stream
    _generate_adapters["ai-gateway-v3"] = ai_gw_generate
    _stream_adapters["openai"] = openai_stream
    _stream_adapters["anthropic"] = anthropic_stream


def register_stream(adapter: str, fn: StreamFn) -> None:
    """Register a stream adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _stream_adapters[adapter] = fn


def register_generate(adapter: str, fn: GenerateFn) -> None:
    """Register a generate adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _generate_adapters[adapter] = fn


# ---------------------------------------------------------------------------
# Connection-check registry — maps *provider* string → check function.
# Keyed by provider (not adapter) because the check verifies "can this
# client reach this provider and does this model exist there".
# ---------------------------------------------------------------------------

_check_fns: dict[str, CheckConnFn] = {}
_check_fns_loaded = False


def _ensure_check_fns() -> None:
    """Lazily register built-in check functions on first call."""
    global _check_fns_loaded  # noqa: PLW0603
    if _check_fns_loaded:
        return
    _check_fns_loaded = True

    from .ai_gateway import check as ai_gw_check
    from .anthropic import check as anthropic_check
    from .openai import check as openai_check

    _check_fns["ai-gateway"] = ai_gw_check.check
    _check_fns["anthropic"] = anthropic_check.check
    _check_fns["openai"] = openai_check.check


def register_check(provider: str, fn: CheckConnFn) -> None:
    """Register a connection-check function for a provider.

    Use this to add checks for custom providers.
    """
    _check_fns[provider] = fn


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


class StreamResult:
    """Wrapper around a message stream. Async-iterable; collects the final result.

    Properties like ``.text`` and ``.tool_calls`` delegate to the final
    ``Message`` snapshot and are available after iteration completes.

    Satisfies :class:`~ai.types.StreamResultLike`.
    """

    def __init__(self, gen: AsyncGenerator[messages_.Message]) -> None:
        self._gen = gen
        self._final: messages_.Message | None = None

    @classmethod
    def from_generator(cls, gen: AsyncGenerator[messages_.Message]) -> StreamResult:
        """Create a :class:`StreamResult` from an async generator.

        This is the public API for middleware that needs to transform or
        replace the stream returned by ``wrap_model``::

            async def wrap_model(self, call, next):
                original = await next(call)

                async def _transformed():
                    async for msg in original:
                        yield modify(msg)

                return StreamResult.from_generator(_transformed())
        """
        return cls(gen)

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[messages_.Message]:
        async for msg in self._gen:
            self._final = msg
            yield msg

    @property
    def text(self) -> str:
        return self._final.text if self._final else ""

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._final.tool_calls if self._final else []

    @property
    def usage(self) -> messages_.Usage | None:
        return self._final.usage if self._final else None

    @property
    def output(self) -> Any:
        """Parsed structured output from the final message, if available."""
        return self._final.output if self._final else None


async def stream(
    model: Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> StreamResultLike:
    """Stream an LLM response.

    Returns a :class:`StreamResultLike` that is async-iterable and
    collects the final ``Message``.  After iteration, access ``.text``,
    ``.tool_calls``, ``.usage``, etc.

    Without middleware the concrete type is :class:`StreamResult`; with
    middleware it may be any :class:`~ai.StreamResultLike`.
    """
    call = middleware_.ModelContext(
        model=model,
        messages=messages,
        tools=tools,
        output_type=output_type,
        client=client,
        kwargs=kwargs,
    )

    async def _real(call: middleware_.ModelContext) -> StreamResultLike:
        _ensure_adapters()
        c = call.client or _auto_client(call.model)
        adapter_fn = _stream_adapters.get(call.model.adapter)
        if adapter_fn is None:
            registered = ", ".join(sorted(_stream_adapters)) or "(none)"
            raise KeyError(
                f"No stream adapter registered for adapter={call.model.adapter!r}. "
                f"Registered: {registered}"
            )
        return StreamResult(
            adapter_fn(
                c,
                call.model,
                call.messages,
                tools=call.tools,
                output_type=call.output_type,
                **call.kwargs,
            )
        )

    chain = middleware_._build_model_chain(_real)
    return await chain(call)


async def generate(
    model: Model,
    messages: list[messages_.Message],
    params: GenerateParams | None = None,
    *,
    client: Client | None = None,
) -> messages_.Message:
    """Generate a response (images, video, etc.).

    Resolves the adapter function from ``model.adapter``, auto-creates a
    :class:`Client` from env vars if none is provided.

    ``params`` controls the generation type:

    * :class:`ImageParams` — image generation (``/image-model``).
    * :class:`VideoParams` — video generation (``/video-model``).
    * ``None`` — auto-detect from ``model.capabilities``.
    """
    call = middleware_.GenerateContext(
        model=model,
        messages=messages,
        params=params,
        client=client,
    )

    async def _real(call: middleware_.GenerateContext) -> messages_.Message:
        _ensure_adapters()
        c = call.client or _auto_client(call.model)
        adapter_fn = _generate_adapters.get(call.model.adapter)
        if adapter_fn is None:
            registered = ", ".join(sorted(_generate_adapters)) or "(none)"
            raise KeyError(
                f"No generate adapter registered for adapter={call.model.adapter!r}. "
                f"Registered: {registered}"
            )
        return await adapter_fn(c, call.model, call.messages, params=call.params)

    chain = middleware_._build_generate_chain(_real)
    return await chain(call)


async def check_connection(
    model: Model,
    *,
    client: Client | None = None,
) -> bool:
    """Check whether *client* can reach *model*'s provider and the model exists.

    Returns ``True`` when the credentials are valid **and** the model is
    available on the remote side — i.e. a subsequent :func:`stream` or
    :func:`generate` call should succeed (network conditions permitting).

    This only hits free metadata endpoints; no tokens or credits are
    consumed.

    If no *client* is given, one is auto-created from environment
    variables (same logic as :func:`stream`).

    Non-auth transport errors (network failures, 5xx) are raised rather
    than returning ``False`` so that callers can distinguish "bad
    credentials / unknown model" from "provider unreachable".
    """
    _ensure_check_fns()
    c = client or _auto_client(model)
    check_fn = _check_fns.get(model.provider)
    if check_fn is None:
        registered = ", ".join(sorted(_check_fns)) or "(none)"
        raise KeyError(
            f"No check function registered for provider={model.provider!r}. "
            f"Registered: {registered}"
        )
    return await check_fn(c, model)


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
    "CheckConnFn",
    "Client",
    "GenerateFn",
    "GenerateParams",
    "ImageParams",
    "Model",
    "ModelCost",
    "StreamFn",
    "StreamResult",
    "StreamResultLike",
    "VideoParams",
    # Catalog
    "get_models",
    "get_providers",
    "model",
    "register_catalog",
    # Public API
    "buffer",
    "check_connection",
    "generate",
    "register_check",
    "register_generate",
    "register_stream",
    "stream",
]
