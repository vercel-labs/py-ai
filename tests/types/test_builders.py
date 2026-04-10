"""Tests for message builder helpers."""

from __future__ import annotations

import pytest

from ai.types import builders, messages

# -- system_message --------------------------------------------------------


def test_system_message_from_string() -> None:
    msg = builders.system_message("You are helpful.")
    assert msg.role == "system"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], messages.TextPart)
    assert msg.parts[0].text == "You are helpful."


def test_system_message_empty() -> None:
    msg = builders.system_message()
    assert msg.role == "system"
    assert msg.parts == []


# -- user_message ----------------------------------------------------------


def test_user_message_single_string() -> None:
    msg = builders.user_message("Hello")
    assert msg.role == "user"
    assert len(msg.parts) == 1
    assert msg.parts[0].text == "Hello"  # type: ignore[union-attr]


def test_user_message_multiple_strings_stay_separate() -> None:
    msg = builders.user_message("foo", "bar")
    assert len(msg.parts) == 2
    assert msg.parts[0].text == "foo"  # type: ignore[union-attr]
    assert msg.parts[1].text == "bar"  # type: ignore[union-attr]


def test_user_message_mixed_content() -> None:
    fp = messages.FilePart(data="https://example.com/img.png", media_type="image/png")
    msg = builders.user_message("Describe this:", fp, "Thanks")
    assert len(msg.parts) == 3
    assert isinstance(msg.parts[0], messages.TextPart)
    assert isinstance(msg.parts[1], messages.FilePart)
    assert isinstance(msg.parts[2], messages.TextPart)


def test_user_message_part_passthrough() -> None:
    tp = messages.TextPart(text="already a part")
    msg = builders.user_message(tp)
    assert msg.parts[0] is tp


# -- assistant_message -----------------------------------------------------


def test_assistant_message_with_thinking() -> None:
    r = builders.thinking("hmm let me think")
    msg = builders.assistant_message(r, "Here's my answer.")
    assert msg.role == "assistant"
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[0], messages.ReasoningPart)
    assert isinstance(msg.parts[1], messages.TextPart)


def test_assistant_message_with_tool_call_part() -> None:
    tool = messages.ToolCallPart(tool_call_id="tc-1", tool_name="test", tool_args="{}")
    msg = builders.assistant_message("calling tool", tool)
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[1], messages.ToolCallPart)


# -- file_part -------------------------------------------------------------


def test_file_part_from_url() -> None:
    fp = builders.file_part("https://example.com/image.png")
    assert isinstance(fp, messages.FilePart)
    assert fp.data == "https://example.com/image.png"
    assert fp.media_type == "image/png"


def test_file_part_from_bytes_with_explicit_media_type() -> None:
    fp = builders.file_part(b"\x00\x00", media_type="application/octet-stream")
    assert isinstance(fp, messages.FilePart)
    assert fp.media_type == "application/octet-stream"


def test_file_part_from_bytes_auto_detect_png() -> None:
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    fp = builders.file_part(png_header)
    assert fp.media_type == "image/png"


def test_file_part_from_bytes_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot detect media_type"):
        builders.file_part(b"\x00\x00\x00")


def test_file_part_with_filename() -> None:
    fp = builders.file_part(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, filename="cat.png")
    assert fp.filename == "cat.png"


# -- thinking --------------------------------------------------------------


def test_thinking_basic() -> None:
    r = builders.thinking("deep thoughts")
    assert isinstance(r, messages.ReasoningPart)
    assert r.text == "deep thoughts"
    assert r.signature is None


def test_thinking_with_signature() -> None:
    r = builders.thinking("deep thoughts", signature="sig123")
    assert r.signature == "sig123"


# -- tool_message ----------------------------------------------------------


def test_tool_message_merges_tool_messages() -> None:
    m1 = messages.Message(
        role="tool",
        parts=[builders.tool_result("tc-1", result=1, tool_name="a")],
    )
    m2 = messages.Message(
        role="tool",
        parts=[builders.tool_result("tc-2", result=2, tool_name="b")],
    )

    merged = builders.tool_message(m1, m2)

    assert merged.role == "tool"
    assert [part.tool_call_id for part in merged.tool_results] == ["tc-1", "tc-2"]


def test_tool_message_accepts_message_list() -> None:
    msgs = [
        messages.Message(
            role="tool",
            parts=[builders.tool_result("tc-1", result=1, tool_name="a")],
        ),
        messages.Message(
            role="tool",
            parts=[builders.tool_result("tc-2", result=2, tool_name="b")],
        ),
    ]

    merged = builders.tool_message(msgs)

    assert [part.tool_call_id for part in merged.tool_results] == ["tc-1", "tc-2"]


def test_tool_message_wraps_tool_result_parts() -> None:
    merged = builders.tool_message(
        builders.tool_result("tc-1", result=1, tool_name="a"),
        builders.tool_result("tc-2", result=2, tool_name="b"),
    )

    assert merged.role == "tool"
    assert [part.tool_call_id for part in merged.tool_results] == ["tc-1", "tc-2"]


def test_tool_message_builds_from_keyword_fields() -> None:
    message = builders.tool_message(tool_call_id="tc-1", result=1, tool_name="weather")

    assert message.role == "tool"
    assert len(message.tool_results) == 1
    assert message.tool_results[0].tool_call_id == "tc-1"
    assert message.tool_results[0].tool_name == "weather"
    assert message.tool_results[0].result == 1
    assert message.tool_results[0].is_error is False


def test_tool_message_builds_error_result_from_keyword_fields() -> None:
    message = builders.tool_message(
        tool_call_id="tc-1",
        result="Denied",
        tool_name="weather",
        is_error=True,
    )

    assert message.tool_results[0].is_error is True
    assert message.tool_results[0].result == "Denied"


def test_tool_message_rejects_non_tool_message() -> None:
    with pytest.raises(TypeError, match="Expected tool message"):
        builders.tool_message(builders.user_message("hello"))


def test_tool_message_rejects_non_result_parts() -> None:
    invalid = messages.Message(role="tool", parts=[messages.TextPart(text="bad")])
    with pytest.raises(TypeError, match="ToolResultPart"):
        builders.tool_message(invalid)


# -- type coercion edge cases ----------------------------------------------


def test_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        builders.user_message(42)  # type: ignore[arg-type]
