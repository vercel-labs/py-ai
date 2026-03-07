"""Vercel AI Gateway provider using the v3 protocol.

Communicates directly with the gateway at ``/v3/ai/language-model``
using the AI SDK's native ``LanguageModelV3`` wire format.  The gateway
server handles translation to each provider's API.

Usage::

    import vercel_ai_sdk as ai

    llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-sonnet-4")

    # or with custom settings
    llm = ai.ai_gateway.GatewayModel(
        model="openai/gpt-4.1",
        api_key="sk-...",
        provider_options={"gateway": {"order": ["bedrock", "openai"]}},
    )
"""

import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import httpx
import pydantic

from .. import core
from . import errors as errors_
from . import protocol as protocol_

_DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_PROTOCOL_VERSION = "0.0.1"


class GatewayModel(core.llm.LanguageModel):
    """Vercel AI Gateway language model using the v3 protocol.

    Sends the AI SDK's native message format directly to the gateway
    server and receives responses in the AI SDK's native stream-part
    format.  The gateway server handles all provider-specific
    translation.

    Args:
        model: Model identifier in ``provider/model`` format
            (e.g. ``'anthropic/claude-sonnet-4'``).
        api_key: API key.  Falls back to ``AI_GATEWAY_API_KEY``.
        base_url: Gateway base URL.
        provider_options: Gateway options (``order``, ``only``,
            ``models``, ``byok``, ``tags``, etc.).
        headers: Extra headers for every request.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        provider_options: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY") or ""
        self._base_url = base_url.rstrip("/")
        self._provider_options = provider_options
        self._extra_headers = headers or {}
        self._transport = _transport

    # -- Internals -----------------------------------------------------------

    def _headers(self, *, streaming: bool) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "ai-gateway-protocol-version": _PROTOCOL_VERSION,
            "ai-language-model-specification-version": "3",
            "ai-language-model-id": self._model,
            "ai-language-model-streaming": str(streaming).lower(),
        }
        if self._api_key:
            h["ai-gateway-auth-method"] = "api-key"
        h.update(self._extra_headers)
        return h

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise a typed :class:`GatewayError` for HTTP >= 400."""
        try:
            body: Any = response.json()
        except Exception:
            body = response.text
        raise errors_.create_gateway_error(
            response_body=body,
            status_code=response.status_code,
            api_key_provided=bool(self._api_key),
        )

    # -- Stream events -------------------------------------------------------

    async def stream_events(
        self,
        messages: list[core.messages.Message],
        tools: Sequence[core.tools.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[core.llm.StreamEvent]:
        """Yield ``StreamEvent`` objects from the gateway SSE stream."""
        body = await protocol_.build_request_body(
            messages,
            tools=tools,
            output_type=output_type,
            provider_options=self._provider_options,
        )
        url = f"{self._base_url}/language-model"
        try:
            async with (
                httpx.AsyncClient(transport=self._transport) as client,
                client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=self._headers(streaming=True),
                    timeout=httpx.Timeout(timeout=300.0, connect=10.0),
                ) as response,
            ):
                if response.status_code >= 400:
                    await response.aread()
                    await self._raise_for_status(response)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for event in protocol_.parse_stream_part(data):
                        yield event

        except errors_.GatewayError:
            raise
        except httpx.TimeoutException as exc:
            raise errors_.GatewayTimeoutError(
                cause=exc,
            ) from exc
        except Exception as exc:
            raise errors_.GatewayResponseError(
                message=(
                    f"Invalid error response format: Gateway request failed: {exc}"
                ),
                cause=exc,
            ) from exc

    # -- LanguageModel interface ---------------------------------------------

    @override
    async def stream(
        self,
        messages: list[core.messages.Message],
        tools: Sequence[core.tools.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[core.messages.Message]:
        handler = core.llm.StreamHandler()
        msg: core.messages.Message | None = None
        async for event in self.stream_events(messages, tools, output_type):
            msg = handler.handle_event(event)
            yield msg

        if output_type is not None and msg is not None and msg.text:
            data = json.loads(msg.text)
            output_type.model_validate(data)
            part = core.messages.StructuredOutputPart(
                data=data,
                output_type_name=(
                    f"{output_type.__module__}.{output_type.__qualname__}"
                ),
            )
            msg = msg.model_copy()
            msg.parts = [*msg.parts, part]
            yield msg


# ---------------------------------------------------------------------------
# Stubs for future model types
# ---------------------------------------------------------------------------


class GatewayEmbeddingModel:
    """Stub — not yet implemented."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        raise NotImplementedError("GatewayEmbeddingModel is not yet implemented.")


class GatewayImageModel:
    """Stub — not yet implemented."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        raise NotImplementedError("GatewayImageModel is not yet implemented.")


class GatewayVideoModel:
    """Stub — not yet implemented."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        raise NotImplementedError("GatewayVideoModel is not yet implemented.")
