"""Tests for MediaModel: extraction and message assembly."""

from __future__ import annotations

from typing import Any

import pytest

from vercel_ai_sdk.core.media import MediaModel, MediaResult
from vercel_ai_sdk.core.messages import FilePart, Message, TextPart, Usage

# ---------------------------------------------------------------------------
# Concrete stub for testing the base class
# ---------------------------------------------------------------------------


class _StubMediaModel(MediaModel):
    """Minimal concrete implementation that just returns what we tell it to."""

    def __init__(self, result: MediaResult) -> None:
        self._result = result

    async def make_request(
        self,
        prompt: str,
        input_files: list[FilePart],
        *,
        n: int = 1,
        provider_options: dict[str, Any] | None = None,
    ) -> MediaResult:
        return self._result


# ---------------------------------------------------------------------------
# _extract_prompt
# ---------------------------------------------------------------------------


class TestExtractPrompt:
    def test_user_text(self) -> None:
        msgs = [Message(role="user", parts=[TextPart(text="hello world")])]
        assert MediaModel._extract_prompt(msgs) == "hello world"

    def test_system_and_user(self) -> None:
        msgs = [
            Message(role="system", parts=[TextPart(text="be helpful")]),
            Message(role="user", parts=[TextPart(text="draw a cat")]),
        ]
        assert MediaModel._extract_prompt(msgs) == "be helpful draw a cat"

    def test_ignores_assistant(self) -> None:
        msgs = [
            Message(role="user", parts=[TextPart(text="hello")]),
            Message(role="assistant", parts=[TextPart(text="ignored")]),
        ]
        assert MediaModel._extract_prompt(msgs) == "hello"

    def test_multiple_text_parts(self) -> None:
        msgs = [
            Message(
                role="user",
                parts=[TextPart(text="first"), TextPart(text="second")],
            )
        ]
        assert MediaModel._extract_prompt(msgs) == "first second"

    def test_skips_non_text_parts(self) -> None:
        msgs = [
            Message(
                role="user",
                parts=[
                    TextPart(text="prompt"),
                    FilePart(data=b"\x89PNG", media_type="image/png"),
                ],
            )
        ]
        assert MediaModel._extract_prompt(msgs) == "prompt"

    def test_empty_messages(self) -> None:
        assert MediaModel._extract_prompt([]) == ""


# ---------------------------------------------------------------------------
# _extract_input_files
# ---------------------------------------------------------------------------


class TestExtractInputFiles:
    def test_user_file_parts(self) -> None:
        img = FilePart(data=b"\x89PNG", media_type="image/png")
        pdf = FilePart(data=b"%PDF", media_type="application/pdf")
        msgs = [Message(role="user", parts=[TextPart(text="hi"), img, pdf])]
        result = MediaModel._extract_input_files(msgs)
        assert result == [img, pdf]

    def test_ignores_assistant_files(self) -> None:
        img = FilePart(data=b"\x89PNG", media_type="image/png")
        msgs = [Message(role="assistant", parts=[img])]
        assert MediaModel._extract_input_files(msgs) == []

    def test_ignores_system_files(self) -> None:
        img = FilePart(data=b"\x89PNG", media_type="image/png")
        msgs = [Message(role="system", parts=[img])]
        assert MediaModel._extract_input_files(msgs) == []

    def test_returns_all_media_types(self) -> None:
        """Unlike the old extract_input_images, this returns ALL file parts."""
        img = FilePart(data=b"\x89PNG", media_type="image/png")
        audio = FilePart(data=b"\xff\xfb", media_type="audio/mpeg")
        video = FilePart(data=b"\x00\x00", media_type="video/mp4")
        msgs = [Message(role="user", parts=[img, audio, video])]
        result = MediaModel._extract_input_files(msgs)
        assert len(result) == 3

    def test_empty_messages(self) -> None:
        assert MediaModel._extract_input_files([]) == []

    def test_multiple_user_messages(self) -> None:
        img1 = FilePart(data=b"\x89PNG", media_type="image/png")
        img2 = FilePart(data=b"\xff\xd8", media_type="image/jpeg")
        msgs = [
            Message(role="user", parts=[img1]),
            Message(role="user", parts=[img2]),
        ]
        result = MediaModel._extract_input_files(msgs)
        assert result == [img1, img2]


# ---------------------------------------------------------------------------
# _build_message
# ---------------------------------------------------------------------------


class TestBuildMessage:
    def test_wraps_files_in_message(self) -> None:
        fp = FilePart(data=b"\x89PNG", media_type="image/png")
        result = MediaResult(files=[fp])
        msg = MediaModel._build_message(result)
        assert msg.role == "assistant"
        assert len(msg.parts) == 1
        assert msg.images[0] is fp

    def test_includes_usage(self) -> None:
        fp = FilePart(data=b"\x89PNG", media_type="image/png")
        usage = Usage(input_tokens=10, output_tokens=20)
        result = MediaResult(files=[fp], usage=usage)
        msg = MediaModel._build_message(result)
        assert msg.usage is not None
        assert msg.usage.input_tokens == 10
        assert msg.usage.output_tokens == 20

    def test_no_usage(self) -> None:
        result = MediaResult(files=[])
        msg = MediaModel._build_message(result)
        assert msg.usage is None

    def test_empty_files(self) -> None:
        result = MediaResult(files=[])
        msg = MediaModel._build_message(result)
        assert msg.parts == []


# ---------------------------------------------------------------------------
# Integration: generate() calls make_request() and wraps result
# ---------------------------------------------------------------------------


class TestGenerateIntegration:
    @pytest.mark.asyncio
    async def test_generate_round_trip(self) -> None:
        """The base class extracts prompt/files and wraps the result."""
        fp_out = FilePart(data="b64data", media_type="image/png")
        usage = Usage(input_tokens=5, output_tokens=15)
        stub = _StubMediaModel(MediaResult(files=[fp_out], usage=usage))

        # We can't call generate() directly on MediaModel since it doesn't
        # define one — subclasses do. But we can verify the pipeline by
        # calling the helpers manually.
        prompt = stub._extract_prompt(
            [Message(role="user", parts=[TextPart(text="a sunset")])]
        )
        assert prompt == "a sunset"

        input_files = stub._extract_input_files(
            [
                Message(
                    role="user",
                    parts=[FilePart(data=b"\x89PNG", media_type="image/png")],
                )
            ]
        )
        assert len(input_files) == 1

        result = await stub.make_request(prompt, input_files)
        msg = stub._build_message(result)
        assert msg.role == "assistant"
        assert msg.images == [fp_out]
        assert msg.usage == usage
