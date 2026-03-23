"""Vercel AI Gateway video model."""

from __future__ import annotations

import json
import os
from typing import Any, override

import httpx

from ...types import messages as messages_
from ..core import video as video_
from ..core.media import base as media_base
from ..core.media import detect as detect_media_type
from ..core.media import download as media_download
from . import errors as errors_
from .llm import _DEFAULT_BASE_URL, _base_headers, _file_part_to_wire, _raise_for_status


class GatewayVideoModel(video_.VideoModel):
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
    async def make_request(
        self,
        prompt: str,
        input_files: list[messages_.FilePart],
        *,
        n: int = 1,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        duration: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> media_base.MediaResult:
        image_wire: dict[str, Any] | None = None
        if input_files:
            image_wire = _file_part_to_wire(input_files[0])

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
            error_type = event_data.get("errorType", "")
            if status == 400 or error_type == "invalid_request_error":
                raise errors_.GatewayInvalidRequestError(
                    message=message, status_code=status
                )
            raise errors_.GatewayResponseError(message=message, status_code=status)

        # Handle result event
        raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
        files: list[messages_.FilePart] = []
        for video_data in raw_videos:
            file_part = await self._video_data_to_file_part(video_data)
            files.append(file_part)

        return media_base.MediaResult(files=files)

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
    ) -> messages_.FilePart:
        """Convert a gateway video result to a :class:`FilePart`.

        Handles ``{type: "url", url, mediaType}`` (downloads the video)
        and ``{type: "base64", data, mediaType}``.
        """
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            video_url = video_data["url"]
            downloaded_bytes, content_type = await media_download.download(video_url)
            # Prefer provider mediaType, then download content-type, then detect
            if media_type == "video/mp4" and content_type:
                media_type = content_type
            detected = detect_media_type.detect_media_type(
                downloaded_bytes, detect_media_type.VIDEO_SIGNATURES
            )
            if detected:
                media_type = detected
            return messages_.FilePart(
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
        return messages_.FilePart(
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
