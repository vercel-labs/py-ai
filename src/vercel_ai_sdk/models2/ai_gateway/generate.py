"""AI Gateway v3 generation adapter — image-model and video-model endpoints.

Provides typed parameter objects (:class:`ImageParams`, :class:`VideoParams`)
and a unified :func:`generate` entry point that dispatches based on param type
and validates against model capabilities.
"""

from __future__ import annotations

from typing import Any

import httpx
import pydantic

from ...types import messages as messages_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import media as media_
from . import _common

# ---------------------------------------------------------------------------
# Parameter types
# ---------------------------------------------------------------------------

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class ImageParams(pydantic.BaseModel):
    """Parameters for image generation (``/image-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    size: str | None = None
    aspect_ratio: str | None = pydantic.Field(None, alias="aspectRatio")
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, alias="providerOptions"
    )


class VideoParams(pydantic.BaseModel):
    """Parameters for video generation (``/video-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    aspect_ratio: str | None = pydantic.Field(None, alias="aspectRatio")
    resolution: str | None = None
    duration: int | None = None
    fps: int | None = None
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, alias="providerOptions"
    )


GenerateParams = ImageParams | VideoParams


# ---------------------------------------------------------------------------
# Image generation — /image-model
# ---------------------------------------------------------------------------


async def _generate_image(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: ImageParams,
) -> messages_.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _common.extract_prompt(messages)
    input_files = _common.extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["files"] = [_common.file_part_to_wire(f) for f in input_files]

    url = f"{client.base_url.rstrip('/')}/image-model"
    headers = _common.request_headers(client, model, model_type="image")

    response = await client.http.post(url, json=body, headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(
            f"AI Gateway image-model returned HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()
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
        media_type = media_.detect_image_media_type(img_b64) or "image/png"
        files.append(messages_.FilePart(data=img_b64, media_type=media_type))

    return messages_.Message(role="assistant", parts=files, usage=usage)


# ---------------------------------------------------------------------------
# Video generation — /video-model (SSE response)
# ---------------------------------------------------------------------------


async def _generate_video(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: VideoParams,
) -> messages_.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _common.extract_prompt(messages)
    input_files = _common.extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["image"] = _common.file_part_to_wire(input_files[0])

    url = f"{client.base_url.rstrip('/')}/video-model"
    headers = _common.request_headers(client, model, model_type="video")
    headers["accept"] = "text/event-stream"

    async with client.http.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise RuntimeError(
                f"AI Gateway video-model returned HTTP {response.status_code}: "
                f"{response.text}"
            )

        # Read first SSE data event — the gateway sends a single result event.
        event_data: dict[str, Any] = {}
        async for parsed in _common.parse_sse_lines(response):
            event_data = parsed
            break

    if event_data.get("type") == "error":
        raise RuntimeError(
            f"AI Gateway video generation error: "
            f"{event_data.get('message', 'unknown error')}"
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    files: list[messages_.FilePart] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await media_.download(video_data["url"])
            if content_type:
                media_type = content_type
            files.append(
                messages_.FilePart(data=downloaded_bytes, media_type=media_type)
            )
        else:
            raw_data = video_data.get("data", "")
            files.append(messages_.FilePart(data=raw_data, media_type=media_type))

    return messages_.Message(role="assistant", parts=files)


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


def _check_capabilities(
    model: model_.Model,
    params: GenerateParams,
) -> None:
    """Validate that model capabilities match the requested generation type."""
    caps = model.capabilities

    if isinstance(params, VideoParams):
        if "video" not in caps:
            raise ValueError(
                f"Model {model.id!r} does not have 'video' capability "
                f"(capabilities={caps}). Use ImageParams for image models."
            )
        if "text" in caps and "video" not in caps:
            raise ValueError(
                f"Model {model.id!r} is a text model (capabilities={caps}). "
                f"Use stream() for text generation, not generate()."
            )
    elif isinstance(params, ImageParams):
        if "video" in caps and "image" not in caps:
            raise ValueError(
                f"Model {model.id!r} has 'video' capability but not 'image' "
                f"(capabilities={caps}). Use VideoParams for video models."
            )
        if "text" in caps and "image" not in caps:
            raise ValueError(
                f"Model {model.id!r} is a text model (capabilities={caps}). "
                f"Use stream() for text generation, not generate()."
            )


async def generate(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: GenerateParams | None = None,
) -> messages_.Message:
    """Generate media (images or video) through the AI Gateway.

    Dispatches to ``/image-model`` or ``/video-model`` based on ``params``
    type, with fallback to model capabilities when ``params`` is ``None``.

    Raises :class:`ValueError` if the model capabilities don't match the
    requested generation type.
    """
    # Auto-detect from capabilities when no params provided
    if params is None:
        params = VideoParams() if "video" in model.capabilities else ImageParams()

    _check_capabilities(model, params)

    if isinstance(params, VideoParams):
        return await _generate_video(client, model, messages, params)
    return await _generate_image(client, model, messages, params)
