"""Vercel AI Gateway provider using the v3 protocol.

Communicates directly with the gateway using the AI SDK's native wire
formats.  The gateway server handles all provider-specific translation.

Usage::

    import vercel_ai_sdk as ai

    # Language model
    llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-sonnet-4")

    # Image model
    img = ai.ai_gateway.GatewayImageModel(model="google/imagen-4.0-generate-001")
    msg = await img.generate(ai.make_messages(user="A sunset over Tokyo"))

    # Video model
    vid = ai.ai_gateway.GatewayVideoModel(model="google/veo-3.0-generate-001")
    msg = await vid.generate(ai.make_messages(user="A cat on a beach"))
"""

import base64
import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import httpx
import pydantic

from .. import core
from ..core import image_model as image_model_
from ..core import video_model as video_model_
from ..core.media import data as media_data
from ..core.media import detect_media_type
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


def _file_part_to_wire(part: core.messages.FilePart) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# GatewayImageModel
# ---------------------------------------------------------------------------


class GatewayImageModel(image_model_.ImageModel):
    """Vercel AI Gateway image model.

    Sends requests to ``/v3/ai/image-model`` and returns a :class:`Message`
    with :class:`FilePart`\\s for each generated image.

    Args:
        model: Model identifier (e.g. ``'google/imagen-4.0-generate-001'``).
        api_key: API key.  Falls back to ``AI_GATEWAY_API_KEY``.
        base_url: Gateway base URL.
        headers: Extra headers for every request.
    """

    def __init__(
        self,
        model: str = "google/imagen-4.0-generate-001",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        headers: dict[str, str] | None = None,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY") or ""
        self._base_url = base_url.rstrip("/")
        self._extra_headers = headers or {}
        self._transport = _transport

    def _headers(self) -> dict[str, str]:
        return _base_headers(
            self._api_key,
            {
                "ai-image-model-specification-version": "3",
                "ai-model-id": self._model,
                **self._extra_headers,
            },
        )

    @override
    async def generate(
        self,
        messages: list[core.messages.Message],
        *,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> core.messages.Message:
        prompt = image_model_.extract_prompt(messages)
        input_images = image_model_.extract_input_images(messages)

        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "providerOptions": provider_options or {},
        }
        if size is not None:
            body["size"] = size
        if aspect_ratio is not None:
            body["aspectRatio"] = aspect_ratio
        if seed is not None:
            body["seed"] = seed
        if input_images:
            body["files"] = [_file_part_to_wire(f) for f in input_images]

        url = f"{self._base_url}/image-model"
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                response = await client.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=httpx.Timeout(timeout=300.0, connect=10.0),
                )
                if response.status_code >= 400:
                    await _raise_for_status(response, api_key=self._api_key)

                data = response.json()

        except errors_.GatewayError:
            raise
        except httpx.TimeoutException as exc:
            raise errors_.GatewayTimeoutError(cause=exc) from exc
        except Exception as exc:
            raise errors_.GatewayResponseError(
                message=f"Gateway image request failed: {exc}",
                cause=exc,
            ) from exc

        # Parse response: {images: string[], warnings?, usage?}
        raw_images: list[str] = data.get("images", [])
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = core.messages.Usage(
                input_tokens=usage_data.get("inputTokens") or 0,
                output_tokens=usage_data.get("outputTokens") or 0,
            )

        parts: list[core.messages.Part] = []
        for img_b64 in raw_images:
            # Detect media type from base64 data, default to image/png
            media_type = detect_media_type.detect_image_media_type(img_b64)
            parts.append(
                core.messages.FilePart(
                    data=img_b64,
                    media_type=media_type or "image/png",
                )
            )

        return core.messages.Message(
            role="assistant",
            parts=parts,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# GatewayVideoModel
# ---------------------------------------------------------------------------


class GatewayVideoModel(video_model_.VideoModel):
    """Vercel AI Gateway video model.

    Sends requests to ``/v3/ai/video-model`` (with SSE response) and returns
    a :class:`Message` with :class:`FilePart`\\s for each generated video.

    Args:
        model: Model identifier (e.g. ``'google/veo-3.0-generate-001'``).
        api_key: API key.  Falls back to ``AI_GATEWAY_API_KEY``.
        base_url: Gateway base URL.
        headers: Extra headers for every request.
    """

    def __init__(
        self,
        model: str = "google/veo-3.0-generate-001",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        headers: dict[str, str] | None = None,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY") or ""
        self._base_url = base_url.rstrip("/")
        self._extra_headers = headers or {}
        self._transport = _transport

    def _headers(self) -> dict[str, str]:
        return _base_headers(
            self._api_key,
            {
                "ai-video-model-specification-version": "3",
                "ai-model-id": self._model,
                "accept": "text/event-stream",
                **self._extra_headers,
            },
        )

    @override
    async def generate(
        self,
        messages: list[core.messages.Message],
        *,
        n: int = 1,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        duration: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> core.messages.Message:
        prompt = image_model_.extract_prompt(messages)

        # Extract optional input image for image-to-video
        input_images = image_model_.extract_input_images(messages)
        image_wire: dict[str, Any] | None = None
        if input_images:
            image_wire = _file_part_to_wire(input_images[0])

        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "providerOptions": provider_options or {},
        }
        if aspect_ratio is not None:
            body["aspectRatio"] = aspect_ratio
        if resolution is not None:
            body["resolution"] = resolution
        if duration is not None:
            body["duration"] = duration
        if fps is not None:
            body["fps"] = fps
        if seed is not None:
            body["seed"] = seed
        if image_wire is not None:
            body["image"] = image_wire

        url = f"{self._base_url}/video-model"
        try:
            async with (
                httpx.AsyncClient(transport=self._transport) as client,
                client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=httpx.Timeout(timeout=600.0, connect=10.0),
                ) as response,
            ):
                if response.status_code >= 400:
                    await response.aread()
                    await _raise_for_status(response, api_key=self._api_key)

                # Parse SSE: read the first data event
                event_data = await self._read_first_sse_event(response)

        except errors_.GatewayError:
            raise
        except httpx.TimeoutException as exc:
            raise errors_.GatewayTimeoutError(cause=exc) from exc
        except Exception as exc:
            raise errors_.GatewayResponseError(
                message=f"Gateway video request failed: {exc}",
                cause=exc,
            ) from exc

        # Handle error event
        if event_data.get("type") == "error":
            status = event_data.get("statusCode", 500)
            message = event_data.get("message", "Video generation failed")
            # Map to the correct error type based on the status code
            error_type = event_data.get("errorType", "")
            if status == 400 or error_type == "invalid_request_error":
                raise errors_.GatewayInvalidRequestError(
                    message=message, status_code=status
                )
            raise errors_.GatewayResponseError(message=message, status_code=status)

        # Handle result event
        raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
        parts: list[core.messages.Part] = []
        for video_data in raw_videos:
            part = await self._video_data_to_file_part(video_data)
            parts.append(part)

        return core.messages.Message(
            role="assistant",
            parts=parts,
        )

    @staticmethod
    async def _read_first_sse_event(response: httpx.Response) -> dict[str, Any]:
        """Read and parse the first SSE data event from the response."""
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                result: dict[str, Any] = json.loads(payload)
                return result
            except json.JSONDecodeError:
                continue
        raise errors_.GatewayResponseError(
            message="SSE stream ended without a data event",
        )

    @staticmethod
    async def _video_data_to_file_part(
        video_data: dict[str, Any],
    ) -> core.messages.FilePart:
        """Convert a gateway video result to a :class:`FilePart`.

        Handles ``{type: "url", url, mediaType}`` (downloads the video)
        and ``{type: "base64", data, mediaType}``.
        """
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            video_url = video_data["url"]
            downloaded_bytes, content_type = await core.media.download.download(
                video_url
            )
            # Prefer provider mediaType, then download content-type, then detect
            if media_type == "video/mp4" and content_type:
                media_type = content_type
            detected = detect_media_type.detect_media_type(
                downloaded_bytes, detect_media_type.VIDEO_SIGNATURES
            )
            if detected:
                media_type = detected
            return core.messages.FilePart(
                data=downloaded_bytes,
                media_type=media_type,
            )

        # base64
        data = video_data.get("data", "")
        detected = detect_media_type.detect_media_type(
            data, detect_media_type.VIDEO_SIGNATURES
        )
        if detected:
            media_type = detected
        return core.messages.FilePart(
            data=data,
            media_type=media_type,
        )


# ---------------------------------------------------------------------------
# Stubs for future model types
# ---------------------------------------------------------------------------


class GatewayEmbeddingModel:
    """Stub -- not yet implemented."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        raise NotImplementedError("GatewayEmbeddingModel is not yet implemented.")
