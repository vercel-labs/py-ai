"""Vercel AI Gateway image model."""

from __future__ import annotations

import os
from typing import Any, override

import httpx

from ...types import messages as messages_
from ..core import image as image_
from ..core.media import base as media_base
from ..core.media import detect as detect_media_type
from . import errors as errors_
from .llm import _DEFAULT_BASE_URL, _base_headers, _file_part_to_wire, _raise_for_status


class GatewayImageModel(image_.ImageModel):
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
    async def make_request(
        self,
        prompt: str,
        input_files: list[messages_.FilePart],
        *,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> media_base.MediaResult:
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
        if input_files:
            body["files"] = [_file_part_to_wire(f) for f in input_files]

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
            usage = messages_.Usage(
                input_tokens=usage_data.get("inputTokens") or 0,
                output_tokens=usage_data.get("outputTokens") or 0,
            )

        files: list[messages_.FilePart] = []
        for img_b64 in raw_images:
            media_type = detect_media_type.detect_image_media_type(img_b64)
            files.append(
                messages_.FilePart(
                    data=img_b64,
                    media_type=media_type or "image/png",
                )
            )

        return media_base.MediaResult(files=files, usage=usage)
