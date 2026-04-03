"""Integration tests for the AI Gateway v3 video generation adapter.

Every test exercises the real ``generate()`` function with a ``Client``
wired to an ``httpx.MockTransport``, so the full production code path
is covered:

    generate(client, model, messages, VideoParams(...))
      → extract prompt/image from messages
      → httpx POST (mock) to /video-model with SSE accept
      → SSE event parsing
      → video data handling (base64 or URL download)
      → return Message with FileParts
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vercel_ai_sdk.models.ai_gateway import errors
from vercel_ai_sdk.models.ai_gateway.generate import (
    VideoParams,
    generate,
)
from vercel_ai_sdk.models.core import client as client_
from vercel_ai_sdk.models.core import model as model_
from vercel_ai_sdk.types import messages

# MP4 magic bytes (ftyp box)
_MP4_HEADER = bytes(
    [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D]
)
_MP4_B64 = base64.b64encode(_MP4_HEADER).decode()

# WebM magic bytes
_WEBM_HEADER = bytes([0x1A, 0x45, 0xDF, 0xA3])
_WEBM_B64 = base64.b64encode(_WEBM_HEADER).decode()

_VIDEO_MODEL = model_.Model(
    id="google/veo-3.0-generate-001",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
    capabilities=("video",),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(*events: dict[str, Any]) -> str:
    """Build SSE response text from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def _client(
    handler: httpx.MockTransport, *, api_key: str = "test-key"
) -> client_.Client:
    c = client_.Client(base_url="https://gw.test/v3/ai", api_key=api_key)
    c._http = httpx.AsyncClient(transport=handler)
    return c


def _user(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_basic_video_generation_base64(self) -> None:
        """Simple prompt -> one MP4 video back via base64."""
        body = _sse(
            {
                "type": "result",
                "videos": [
                    {"type": "base64", "data": _MP4_B64, "mediaType": "video/mp4"}
                ],
            }
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = _client(httpx.MockTransport(handler))
        msg = await generate(
            client,
            _VIDEO_MODEL,
            [_user("A cat walking on a beach")],
            params=VideoParams(),
        )

        assert msg.role == "assistant"
        assert len(msg.videos) == 1
        assert msg.videos[0].data == _MP4_B64
        assert msg.videos[0].media_type == "video/mp4"

    @pytest.mark.asyncio
    async def test_video_generation_url(self) -> None:
        """Video returned as URL -> downloaded automatically."""
        body = _sse(
            {
                "type": "result",
                "videos": [
                    {
                        "type": "url",
                        "url": "https://storage.example.com/video.mp4",
                        "mediaType": "video/mp4",
                    }
                ],
            }
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = _client(httpx.MockTransport(handler))

        with patch(
            "vercel_ai_sdk.models.core.helpers.media.download",
            new_callable=AsyncMock,
            return_value=(_MP4_HEADER, "video/mp4"),
        ) as mock_dl:
            msg = await generate(
                client,
                _VIDEO_MODEL,
                [_user("A sunset timelapse")],
                params=VideoParams(),
            )

        mock_dl.assert_called_once_with("https://storage.example.com/video.mp4")
        assert len(msg.videos) == 1
        assert msg.videos[0].data == _MP4_HEADER
        assert msg.videos[0].media_type == "video/mp4"

    @pytest.mark.asyncio
    async def test_multiple_videos(self) -> None:
        body = _sse(
            {
                "type": "result",
                "videos": [
                    {"type": "base64", "data": _MP4_B64, "mediaType": "video/mp4"},
                    {"type": "base64", "data": _WEBM_B64, "mediaType": "video/webm"},
                ],
            }
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        msg = await generate(
            _client(httpx.MockTransport(handler)),
            _VIDEO_MODEL,
            [_user("Two versions")],
            params=VideoParams(n=2),
        )
        assert len(msg.videos) == 2
        assert msg.videos[0].media_type == "video/mp4"
        assert msg.videos[1].media_type == "video/webm"


# ---------------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "type": "result",
                        "videos": [
                            {
                                "type": "base64",
                                "data": _MP4_B64,
                                "mediaType": "video/mp4",
                            }
                        ],
                    }
                ),
            )

        client = _client(httpx.MockTransport(handler), api_key="sk-test")
        await generate(
            client,
            _VIDEO_MODEL,
            [_user("test")],
            params=VideoParams(),
        )

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-video-model-specification-version"] == "3"
        assert captured["ai-model-id"] == "google/veo-3.0-generate-001"
        assert captured["accept"] == "text/event-stream"
        assert captured["ai-gateway-auth-method"] == "api-key"

    @pytest.mark.asyncio
    async def test_parameters_forwarded(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "type": "result",
                        "videos": [
                            {
                                "type": "base64",
                                "data": _MP4_B64,
                                "mediaType": "video/mp4",
                            }
                        ],
                    }
                ),
            )

        client = _client(httpx.MockTransport(handler))
        await generate(
            client,
            _VIDEO_MODEL,
            [_user("sunset")],
            params=VideoParams(
                n=2,
                aspect_ratio="16:9",
                resolution="1920x1080",
                duration=5,
                fps=30,
                seed=42,
                provider_options={"google": {"enhancePrompt": True}},
            ),
        )

        assert captured_body["prompt"] == "sunset"
        assert captured_body["n"] == 2
        assert captured_body["aspectRatio"] == "16:9"
        assert captured_body["resolution"] == "1920x1080"
        assert captured_body["duration"] == 5
        assert captured_body["fps"] == 30
        assert captured_body["seed"] == 42
        assert captured_body["providerOptions"] == {"google": {"enhancePrompt": True}}

    @pytest.mark.asyncio
    async def test_url_posts_to_video_model_endpoint(self) -> None:
        captured_url: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_url.append(str(req.url))
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "type": "result",
                        "videos": [
                            {
                                "type": "base64",
                                "data": _MP4_B64,
                                "mediaType": "video/mp4",
                            }
                        ],
                    }
                ),
            )

        client = _client(httpx.MockTransport(handler))
        await generate(
            client,
            _VIDEO_MODEL,
            [_user("test")],
            params=VideoParams(),
        )

        assert captured_url[0] == "https://gw.test/v3/ai/video-model"

    @pytest.mark.asyncio
    async def test_image_to_video_input(self) -> None:
        """Image in user message -> image field in request body."""
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "type": "result",
                        "videos": [
                            {
                                "type": "base64",
                                "data": _MP4_B64,
                                "mediaType": "video/mp4",
                            }
                        ],
                    }
                ),
            )

        png_b64 = base64.b64encode(b"\x89PNG").decode()
        user_msg = messages.Message(
            role="user",
            parts=[
                messages.TextPart(text="Animate this"),
                messages.FilePart(data=png_b64, media_type="image/png"),
            ],
        )
        client = _client(httpx.MockTransport(handler))
        await generate(client, _VIDEO_MODEL, [user_msg], params=VideoParams())

        assert captured_body["prompt"] == "Animate this"
        assert "image" in captured_body
        assert captured_body["image"]["type"] == "file"
        assert captured_body["image"]["mediaType"] == "image/png"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_sse_error_event(self) -> None:
        """Gateway returns an SSE error event -> raises."""
        body = _sse(
            {
                "type": "error",
                "message": "Content policy violation",
                "errorType": "content_filter",
                "statusCode": 400,
                "param": None,
            }
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        with pytest.raises(errors.GatewayInvalidRequestError, match="Content policy"):
            await generate(
                _client(httpx.MockTransport(handler)),
                _VIDEO_MODEL,
                [_user("test")],
                params=VideoParams(),
            )

    @pytest.mark.asyncio
    async def test_401_authentication_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Bad key",
                        "type": "authentication_error",
                    }
                },
            )

        with pytest.raises(errors.GatewayAuthenticationError):
            await generate(
                _client(httpx.MockTransport(handler)),
                _VIDEO_MODEL,
                [_user("test")],
                params=VideoParams(),
            )

    @pytest.mark.asyncio
    async def test_empty_sse_stream(self) -> None:
        """SSE stream with no data events -> raises."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="")

        with pytest.raises(errors.GatewayResponseError, match="SSE stream ended"):
            await generate(
                _client(httpx.MockTransport(handler)),
                _VIDEO_MODEL,
                [_user("test")],
                params=VideoParams(),
            )
