"""OpenAI provider: _messages_to_openai multimodal conversion tests."""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from vercel_ai_sdk.core.messages import FilePart, Message, TextPart
from vercel_ai_sdk.openai import _messages_to_openai

# -- text-only (regression) ------------------------------------------------


@pytest.mark.asyncio
async def test_user_text_only_is_plain_string() -> None:
    """Text-only user messages should produce a plain content string, not array."""
    msgs = [Message(role="user", parts=[TextPart(text="Hello")])]
    result = await _messages_to_openai(msgs)
    assert result == [{"role": "user", "content": "Hello"}]


# -- images ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_image_url() -> None:
    """Image URL → OpenAI image_url content part."""
    msgs = [
        Message(
            role="user",
            parts=[
                TextPart(text="What's this?"),
                FilePart(data="https://example.com/cat.jpg", media_type="image/jpeg"),
            ],
        )
    ]
    result = await _messages_to_openai(msgs)
    content = result[0]["content"]
    assert content[0] == {"type": "text", "text": "What's this?"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpg"},
    }


@pytest.mark.asyncio
async def test_user_image_base64() -> None:
    """Base64 image data → OpenAI image_url with data URL."""
    b64 = base64.b64encode(b"\x89PNG").decode()
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b64, media_type="image/png")],
        )
    ]
    result = await _messages_to_openai(msgs)
    content = result[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == f"data:image/png;base64,{b64}"


@pytest.mark.asyncio
async def test_user_image_bytes() -> None:
    """Raw bytes image → OpenAI image_url with data URL."""
    raw = b"\x89PNG"
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=raw, media_type="image/png")],
        )
    ]
    result = await _messages_to_openai(msgs)
    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_user_image_wildcard_becomes_jpeg() -> None:
    """image/* media type is normalized to image/jpeg for the data URL."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data="https://example.com/img", media_type="image/*")],
        )
    ]
    result = await _messages_to_openai(msgs)
    # URL passthrough: no data URL conversion needed
    assert result[0]["content"][0]["image_url"]["url"] == "https://example.com/img"


@pytest.mark.asyncio
async def test_user_image_data_url() -> None:
    """data: URL image → base64 extracted correctly for image_url."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data="data:image/png;base64,AQID", media_type="image/png")],
        )
    ]
    result = await _messages_to_openai(msgs)
    # data: URLs pass through directly for images
    assert result[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AQID"


# -- audio -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_audio_base64() -> None:
    """Audio base64 → OpenAI input_audio part."""
    b64 = base64.b64encode(b"\xff\xfb").decode()
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b64, media_type="audio/wav")],
        )
    ]
    result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"] == b64
    assert part["input_audio"]["format"] == "wav"


@pytest.mark.asyncio
async def test_user_audio_data_url_extracts_base64() -> None:
    """Audio data: URL → base64 prefix stripped for input_audio."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data="data:audio/wav;base64,AAAA", media_type="audio/wav")],
        )
    ]
    result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"] == "AAAA"


@pytest.mark.asyncio
async def test_user_audio_url_downloads() -> None:
    """Audio URLs are auto-downloaded since OpenAI requires base64."""
    fake_audio = b"\xff\xfb\x90\x00"
    msgs = [
        Message(
            role="user",
            parts=[
                FilePart(data="https://example.com/clip.wav", media_type="audio/wav")
            ],
        )
    ]
    with patch(
        "vercel_ai_sdk.core.media.download.download",
        new_callable=AsyncMock,
        return_value=(fake_audio, "audio/wav"),
    ):
        result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    # Should be base64 of the downloaded bytes
    assert part["input_audio"]["data"] == base64.b64encode(fake_audio).decode()


# -- PDF -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_pdf_base64() -> None:
    """PDF base64 → OpenAI file part."""
    b64 = base64.b64encode(b"%PDF-1.4").decode()
    msgs = [
        Message(
            role="user",
            parts=[
                FilePart(data=b64, media_type="application/pdf", filename="report.pdf")
            ],
        )
    ]
    result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "report.pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


@pytest.mark.asyncio
async def test_user_pdf_url_downloads() -> None:
    """PDF URLs are auto-downloaded since OpenAI requires base64."""
    fake_pdf = b"%PDF-1.4 fake content"
    msgs = [
        Message(
            role="user",
            parts=[
                FilePart(
                    data="https://example.com/doc.pdf",
                    media_type="application/pdf",
                    filename="doc.pdf",
                )
            ],
        )
    ]
    with patch(
        "vercel_ai_sdk.core.media.download.download",
        new_callable=AsyncMock,
        return_value=(fake_pdf, "application/pdf"),
    ):
        result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "doc.pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


# -- text/* ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_text_file_bytes() -> None:
    """text/* file with bytes data → decoded to text content part."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b"Hello, world!", media_type="text/plain")],
        )
    ]
    result = await _messages_to_openai(msgs)
    part = result[0]["content"][0]
    assert part == {"type": "text", "text": "Hello, world!"}


# -- unsupported -----------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_media_type_raises() -> None:
    """Unknown media type → ValueError."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b"\x00", media_type="application/octet-stream")],
        )
    ]
    with pytest.raises(ValueError, match="Unsupported media type"):
        await _messages_to_openai(msgs)
