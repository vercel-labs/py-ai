"""Integration tests for ``GatewayImageModel``.

Every test exercises the real ``model.generate()`` method with an injected
``httpx.MockTransport``, so the full production code path is covered:

    model.generate()
      → extract prompt/images from messages
      → httpx POST (mock) to /image-model
      → JSON response parsing
      → media type detection
      → return Message with FileParts
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from vercel_ai_sdk.models.ai_gateway import GatewayImageModel, errors
from vercel_ai_sdk.types import messages

# 1x1 transparent PNG (minimal valid PNG for magic-byte detection)
_PNG_HEADER = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
_PNG_B64 = base64.b64encode(_PNG_HEADER).decode()

# 1x1 JPEG header
_JPEG_HEADER = bytes([0xFF, 0xD8, 0xFF, 0xE0])
_JPEG_B64 = base64.b64encode(_JPEG_HEADER).decode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_model(
    handler: httpx.MockTransport,
    *,
    model: str = "google/imagen-4.0-generate-001",
    api_key: str = "test-key",
) -> GatewayImageModel:
    return GatewayImageModel(
        model=model,
        api_key=api_key,
        base_url="https://gw.test/v3/ai",
        _transport=handler,
    )


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
    async def test_basic_image_generation(self) -> None:
        """Simple prompt → one PNG image back."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"images": [_PNG_B64]},
            )

        model = _image_model(httpx.MockTransport(handler))
        msg = await model.generate([_user("A sunset over Tokyo")])

        assert msg.role == "assistant"
        assert len(msg.images) == 1
        assert msg.images[0].data == _PNG_B64
        assert msg.images[0].media_type == "image/png"

    @pytest.mark.asyncio
    async def test_multiple_images(self) -> None:
        """Request n=3 images."""

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["n"] == 3
            return httpx.Response(
                200,
                json={"images": [_PNG_B64, _JPEG_B64, _PNG_B64]},
            )

        model = _image_model(httpx.MockTransport(handler))
        msg = await model.generate([_user("Three cats")], n=3)

        assert len(msg.images) == 3
        assert msg.images[0].media_type == "image/png"
        assert msg.images[1].media_type == "image/jpeg"
        assert msg.images[2].media_type == "image/png"

    @pytest.mark.asyncio
    async def test_usage_parsing(self) -> None:
        """Usage data from response surfaces on the Message."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "images": [_PNG_B64],
                    "usage": {"inputTokens": 50, "outputTokens": 100},
                },
            )

        model = _image_model(httpx.MockTransport(handler))
        msg = await model.generate([_user("a dog")])

        assert msg.usage is not None
        assert msg.usage.input_tokens == 50
        assert msg.usage.output_tokens == 100


# ---------------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        model = _image_model(
            httpx.MockTransport(handler),
            model="openai/gpt-image-1",
            api_key="sk-test",
        )
        await model.generate([_user("Hi")])

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-image-model-specification-version"] == "3"
        assert captured["ai-model-id"] == "openai/gpt-image-1"
        assert captured["ai-gateway-auth-method"] == "api-key"

    @pytest.mark.asyncio
    async def test_parameters_forwarded(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        model = _image_model(httpx.MockTransport(handler))
        await model.generate(
            [_user("landscape")],
            n=2,
            size="1024x1024",
            aspect_ratio="16:9",
            seed=42,
            provider_options={"google": {"style": "vivid"}},
        )

        assert captured_body["prompt"] == "landscape"
        assert captured_body["n"] == 2
        assert captured_body["size"] == "1024x1024"
        assert captured_body["aspectRatio"] == "16:9"
        assert captured_body["seed"] == 42
        assert captured_body["providerOptions"] == {"google": {"style": "vivid"}}

    @pytest.mark.asyncio
    async def test_input_images_forwarded(self) -> None:
        """Input images from user messages → files in request body."""
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        user_msg = messages.Message(
            role="user",
            parts=[
                messages.TextPart(text="Edit this"),
                messages.FilePart(data=_PNG_B64, media_type="image/png"),
            ],
        )
        model = _image_model(httpx.MockTransport(handler))
        await model.generate([user_msg])

        assert captured_body["prompt"] == "Edit this"
        assert "files" in captured_body
        assert len(captured_body["files"]) == 1
        assert captured_body["files"][0]["type"] == "file"
        assert captured_body["files"][0]["mediaType"] == "image/png"

    @pytest.mark.asyncio
    async def test_url_posts_to_image_model_endpoint(self) -> None:
        captured_url: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_url.append(str(req.url))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        model = _image_model(httpx.MockTransport(handler))
        await model.generate([_user("test")])

        assert captured_url[0] == "https://gw.test/v3/ai/image-model"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_401_authentication_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                    }
                },
            )

        with pytest.raises(errors.GatewayAuthenticationError):
            await _image_model(httpx.MockTransport(handler)).generate([_user("test")])

    @pytest.mark.asyncio
    async def test_429_rate_limit_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limited",
                        "type": "rate_limit_exceeded",
                    }
                },
            )

        with pytest.raises(errors.GatewayRateLimitError):
            await _image_model(httpx.MockTransport(handler)).generate([_user("test")])

    @pytest.mark.asyncio
    async def test_empty_images_returns_empty_message(self) -> None:
        """Gateway returns empty images array → message with no parts."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"images": []})

        msg = await _image_model(httpx.MockTransport(handler)).generate([_user("test")])
        assert len(msg.images) == 0
