"""Anthropic provider: _messages_to_anthropic conversion tests."""

import base64

import pytest

from vercel_ai_sdk.models.anthropic import _messages_to_anthropic
from vercel_ai_sdk.types.messages import FilePart, Message, TextPart, ToolPart

pytestmark = pytest.mark.asyncio


async def test_tool_result_none_still_emits_tool_result() -> None:
    """A tool that returns None must still produce a tool_result block.

    Regression: when part.result is None the converter skipped the tool_result,
    leaving a tool_use without a matching tool_result.  Anthropic rejects this
    with: "tool_use ids were found without tool_result blocks immediately after".
    """
    tool_part = ToolPart(
        tool_call_id="toolu_01abc",
        tool_name="send_notification",
        tool_args="{}",
    )
    tool_part.set_result(None)  # tool returned None (fire-and-forget style)

    messages = [
        Message(role="assistant", parts=[tool_part]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    # Should have: assistant message with tool_use, then user message with tool_result
    assert len(anthropic_msgs) == 2, (
        f"Expected 2 messages (assistant + user/tool_result), "
        f"got {len(anthropic_msgs)}: {anthropic_msgs}"
    )

    assistant_msg = anthropic_msgs[0]
    assert assistant_msg["role"] == "assistant"
    assert any(block["type"] == "tool_use" for block in assistant_msg["content"])

    user_msg = anthropic_msgs[1]
    assert user_msg["role"] == "user"
    tool_results = [b for b in user_msg["content"] if b["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "toolu_01abc"


async def test_tool_with_normal_result() -> None:
    """Baseline: a tool with a normal result produces the correct pair."""
    tool_part = ToolPart(
        tool_call_id="toolu_02xyz",
        tool_name="get_weather",
        tool_args='{"city": "SF"}',
    )
    tool_part.set_result({"temp": 62})

    messages = [
        Message(role="assistant", parts=[tool_part]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[1]["content"][0]["content"] == "{'temp': 62}"


async def test_tool_error_produces_tool_result() -> None:
    """Tool errors must also produce a tool_result block (with is_error=True)."""
    tool_part = ToolPart(
        tool_call_id="toolu_03err",
        tool_name="failing_tool",
        tool_args="{}",
    )
    tool_part.set_error("Connection timeout")

    messages = [
        Message(role="assistant", parts=[tool_part]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    assert len(anthropic_msgs) == 2
    tool_result = anthropic_msgs[1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True
    assert tool_result["content"] == "Connection timeout"


async def test_multiple_tools_one_returns_none() -> None:
    """When one of several tools returns None, all must have tool_results."""
    tool_a = ToolPart(
        tool_call_id="toolu_a",
        tool_name="tool_a",
        tool_args="{}",
    )
    tool_a.set_result("some result")

    tool_b = ToolPart(
        tool_call_id="toolu_b",
        tool_name="tool_b",
        tool_args="{}",
    )
    tool_b.set_result(None)  # returns None

    messages = [
        Message(role="assistant", parts=[tool_a, tool_b]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    assert len(anthropic_msgs) == 2

    # Both tool_use blocks in assistant message
    tool_uses = [b for b in anthropic_msgs[0]["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 2

    # Both tool_result blocks in user message
    tool_results = [
        b for b in anthropic_msgs[1]["content"] if b["type"] == "tool_result"
    ]
    assert len(tool_results) == 2

    result_ids = {r["tool_use_id"] for r in tool_results}
    assert result_ids == {"toolu_a", "toolu_b"}


# -- Multi-turn: consecutive user messages (tool_result + next user) -------


async def test_multi_turn_no_consecutive_same_role_messages() -> None:
    """Multi-turn with tools must not produce consecutive same-role messages.

    Regression: when a previous assistant turn includes a tool call (with
    result), _messages_to_anthropic emits:
      [assistant(tool_use)] [user(tool_result)] [user(next question)]
    The two consecutive user messages violate Anthropic's alternating-role
    requirement, causing: "tool_use ids were found without tool_result
    blocks immediately after".

    The tool_result user message must be merged with the following user
    message (or otherwise avoid consecutive same-role messages).
    """
    tool = ToolPart(
        tool_call_id="toolu_01abc",
        tool_name="talk_to_mothership",
        tool_args='{"question": "when?"}',
    )
    tool.set_result({"value": "Soon."})

    messages = [
        Message(role="user", parts=[TextPart(text="when will the robots take over?")]),
        Message(
            role="assistant",
            parts=[
                TextPart(text="I'll check with the mothership."),
                tool,
                TextPart(text="The mothership has spoken: Soon."),
            ],
        ),
        Message(
            role="user",
            parts=[TextPart(text="can you remember the first turn?")],
        ),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    # Verify no consecutive same-role messages
    for i in range(1, len(anthropic_msgs)):
        assert anthropic_msgs[i]["role"] != anthropic_msgs[i - 1]["role"], (
            f"Consecutive same-role messages at indices {i - 1} and {i}: "
            f"both are '{anthropic_msgs[i]['role']}'. "
            f"Full messages: {anthropic_msgs}"
        )


async def test_multi_turn_tool_result_before_user_merged() -> None:
    """When tool_result (user) is followed by a user message, they merge.

    The merged user message should contain both the tool_result blocks
    and the text content from the following user message.
    """
    tool = ToolPart(
        tool_call_id="toolu_01abc",
        tool_name="get_weather",
        tool_args='{"city": "SF"}',
    )
    tool.set_result("Sunny, 62F")

    messages = [
        Message(role="user", parts=[TextPart(text="what's the weather?")]),
        Message(role="assistant", parts=[tool]),
        Message(role="user", parts=[TextPart(text="thanks, what about tomorrow?")]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    # Should be: user, assistant, user (tool_result + text)
    assert len(anthropic_msgs) == 3
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[1]["role"] == "assistant"
    assert anthropic_msgs[2]["role"] == "user"

    # The merged user message should contain the tool_result
    user_content = anthropic_msgs[2]["content"]
    assert isinstance(user_content, list)
    tool_results = [b for b in user_content if b.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "toolu_01abc"


async def test_stream_loop_second_iteration_messages() -> None:
    """Simulates what stream_loop sends on the 2nd LLM call in a multi-turn.

    After the first stream_step returns a tool call, stream_loop appends
    the assistant message (now with status=result after execute_tool) and
    calls stream_step again.  The messages must not have consecutive
    same-role entries.
    """
    tool = ToolPart(
        tool_call_id="toolu_01abc",
        tool_name="talk_to_mothership",
        tool_args='{"question": "test"}',
    )
    tool.set_result("answer")

    # These are the messages that stream_loop would pass to the 2nd stream_step:
    # original user messages + assistant message from 1st step (with tool result)
    messages = [
        Message(role="user", parts=[TextPart(text="ask the mothership")]),
        Message(role="assistant", parts=[tool]),
        # No user message follows — this is the loop, not a new user turn
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    # Should be: user, assistant(tool_use), user(tool_result)
    assert len(anthropic_msgs) == 3
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[1]["role"] == "assistant"
    assert anthropic_msgs[2]["role"] == "user"

    # Verify the tool_result is present
    tool_results = [
        b for b in anthropic_msgs[2]["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1


async def test_pending_tool_does_not_emit_tool_result() -> None:
    """A tool with status='pending' must not produce a tool_result block.

    When stream_step returns a message mid-stream (before tool execution),
    the ToolPart has status='pending'.  The converter must emit only
    the tool_use block — no tool_result.
    """
    tool = ToolPart(
        tool_call_id="toolu_pending",
        tool_name="slow_tool",
        tool_args='{"x": 1}',
    )
    # Don't call set_result — status stays "pending"

    messages = [
        Message(role="user", parts=[TextPart(text="do something")]),
        Message(role="assistant", parts=[tool]),
    ]

    _system, anthropic_msgs = await _messages_to_anthropic(messages)

    # assistant message with tool_use, but NO user message with tool_result
    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in anthropic_msgs[1]["content"])

    # No tool_result anywhere
    for msg in anthropic_msgs:
        if isinstance(msg["content"], list):
            assert not any(b.get("type") == "tool_result" for b in msg["content"])


# -- Multimodal user messages ------------------------------------------------


async def test_user_text_only_is_plain_string() -> None:
    """Text-only user messages should produce a plain content string."""
    msgs = [Message(role="user", parts=[TextPart(text="Hello")])]
    _sys, result = await _messages_to_anthropic(msgs)
    assert result[0]["content"] == "Hello"


async def test_user_image_url() -> None:
    """Image URL → Anthropic image block with url source."""
    msgs = [
        Message(
            role="user",
            parts=[
                TextPart(text="Describe this"),
                FilePart(data="https://example.com/cat.jpg", media_type="image/jpeg"),
            ],
        )
    ]
    _sys, result = await _messages_to_anthropic(msgs)
    content = result[0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
    }


async def test_user_image_base64() -> None:
    """Base64 image → Anthropic image block with base64 source."""
    b64 = base64.b64encode(b"\x89PNG").decode()
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b64, media_type="image/png")],
        )
    ]
    _sys, result = await _messages_to_anthropic(msgs)
    img = result[0]["content"][0]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == b64


async def test_user_pdf_url() -> None:
    """PDF URL → Anthropic document block with url source."""
    msgs = [
        Message(
            role="user",
            parts=[
                FilePart(
                    data="https://example.com/doc.pdf", media_type="application/pdf"
                )
            ],
        )
    ]
    _sys, result = await _messages_to_anthropic(msgs)
    doc = result[0]["content"][0]
    assert doc["type"] == "document"
    assert doc["source"] == {"type": "url", "url": "https://example.com/doc.pdf"}


async def test_user_pdf_base64() -> None:
    """PDF base64 → Anthropic document block with base64 source."""
    b64 = base64.b64encode(b"%PDF-1.4").decode()
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b64, media_type="application/pdf")],
        )
    ]
    _sys, result = await _messages_to_anthropic(msgs)
    doc = result[0]["content"][0]
    assert doc["type"] == "document"
    assert doc["source"]["type"] == "base64"
    assert doc["source"]["media_type"] == "application/pdf"


async def test_user_text_plain_bytes() -> None:
    """text/plain with bytes → Anthropic document with text source."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b"Hello, world!", media_type="text/plain")],
        )
    ]
    _sys, result = await _messages_to_anthropic(msgs)
    doc = result[0]["content"][0]
    assert doc["type"] == "document"
    assert doc["source"]["type"] == "text"
    assert doc["source"]["data"] == "Hello, world!"


async def test_unsupported_media_type_raises() -> None:
    """Unsupported media type → ValueError."""
    msgs = [
        Message(
            role="user",
            parts=[FilePart(data=b"\x00", media_type="video/mp4")],
        )
    ]
    with pytest.raises(ValueError, match="Unsupported media type"):
        await _messages_to_anthropic(msgs)
