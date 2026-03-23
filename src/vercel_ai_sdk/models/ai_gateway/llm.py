"""Vercel AI Gateway language model using the v3 protocol."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import httpx
import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from ..core import llm as llm_
from ..core.media import data as media_data
from . import errors as errors_
from . import protocol as protocol_

_DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_PROTOCOL_VERSION = "0.0.1"


class GatewayModel(llm_.LanguageModel):
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
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[llm_.StreamEvent]:
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
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[messages_.Message]:
        handler = llm_.StreamHandler()
        msg: messages_.Message | None = None
        async for event in self.stream_events(messages, tools, output_type):
            msg = handler.handle_event(event)
            yield msg

        if output_type is not None and msg is not None and msg.text:
            data = json.loads(msg.text)
            output_type.model_validate(data)
            part = messages_.StructuredOutputPart(
                data=data,
                output_type_name=(
                    f"{output_type.__module__}.{output_type.__qualname__}"
                ),
            )
            msg = msg.model_copy()
            msg.parts = [*msg.parts, part]
            yield msg


# ---------------------------------------------------------------------------
# Shared helpers for image/video models
# ---------------------------------------------------------------------------


def _base_headers(api_key: str, extra: dict[str, str]) -> dict[str, str]:
    """Build common gateway headers."""
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
    }
    if api_key:
        h["ai-gateway-auth-method"] = "api-key"
    h.update(extra)
    return h


async def _raise_for_status(response: httpx.Response, *, api_key: str) -> None:
    """Raise a typed :class:`GatewayError` for HTTP >= 400."""
    try:
        body: Any = response.json()
    except Exception:
        body = response.text
    raise errors_.create_gateway_error(
        response_body=body,
        status_code=response.status_code,
        api_key_provided=bool(api_key),
    )


def _file_part_to_wire(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to the gateway wire format for input files."""
    data = part.data
    if isinstance(data, str) and media_data.is_url(data):
        return {"type": "url", "url": data}
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        # Assume raw base64
        b64 = data
    else:
        b64 = str(data)
    return {"type": "file", "data": b64, "mediaType": part.media_type}
