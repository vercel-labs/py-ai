"""StreamHandler: event accumulation, state transitions, message building."""

from vercel_ai_sdk.core.llm import (
    MessageDone,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningStart,
    StreamHandler,
    TextDelta,
    TextEnd,
    TextStart,
    ToolArgsDelta,
    ToolEnd,
    ToolStart,
)
from vercel_ai_sdk.core.messages import ReasoningPart, TextPart, ToolPart

# -- Text streaming --------------------------------------------------------


def test_text_lifecycle() -> None:
    h = StreamHandler(message_id="m1")
    m = h.handle_event(TextStart(block_id="b1"))
    assert len(m.parts) == 1
    part = m.parts[0]
    assert isinstance(part, TextPart)
    assert part.state == "streaming"
    assert part.text == ""

    m = h.handle_event(TextDelta(block_id="b1", delta="Hello"))
    part = m.parts[0]
    assert isinstance(part, TextPart)
    assert part.text == "Hello"
    assert part.delta == "Hello"
    assert part.state == "streaming"

    m = h.handle_event(TextDelta(block_id="b1", delta=" world"))
    part = m.parts[0]
    assert isinstance(part, TextPart)
    assert part.text == "Hello world"
    assert part.delta == " world"

    m = h.handle_event(TextEnd(block_id="b1"))
    part = m.parts[0]
    assert isinstance(part, TextPart)
    assert part.state == "done"
    assert part.delta is None


# -- Reasoning streaming ---------------------------------------------------


def test_reasoning_lifecycle() -> None:
    h = StreamHandler(message_id="m1")
    h.handle_event(ReasoningStart(block_id="r1"))
    m = h.handle_event(ReasoningDelta(block_id="r1", delta="thinking"))
    part = m.parts[0]
    assert isinstance(part, ReasoningPart)
    assert part.text == "thinking"
    assert part.state == "streaming"

    m = h.handle_event(ReasoningEnd(block_id="r1", signature="sig123"))
    part = m.parts[0]
    assert isinstance(part, ReasoningPart)
    assert part.state == "done"
    assert part.signature == "sig123"


# -- Tool streaming --------------------------------------------------------


def test_tool_lifecycle() -> None:
    h = StreamHandler(message_id="m1")
    h.handle_event(ToolStart(tool_call_id="tc1", tool_name="get_weather"))
    m = h.handle_event(ToolArgsDelta(tool_call_id="tc1", delta='{"ci'))
    part = m.parts[0]
    assert isinstance(part, ToolPart)
    assert part.tool_name == "get_weather"
    assert part.tool_args == '{"ci'
    assert part.state == "streaming"
    assert part.args_delta == '{"ci'

    m = h.handle_event(ToolArgsDelta(tool_call_id="tc1", delta='ty":"London"}'))
    part = m.parts[0]
    assert isinstance(part, ToolPart)
    assert part.tool_args == '{"city":"London"}'

    m = h.handle_event(ToolEnd(tool_call_id="tc1"))
    part = m.parts[0]
    assert isinstance(part, ToolPart)
    assert part.state == "done"
    assert part.args_delta is None


# -- Multi-part messages ---------------------------------------------------


def test_reasoning_then_text_then_tool() -> None:
    """Full message: reasoning block, text block, tool call."""
    h = StreamHandler(message_id="m1")
    h.handle_event(ReasoningStart(block_id="r1"))
    h.handle_event(ReasoningDelta(block_id="r1", delta="Let me think"))
    h.handle_event(ReasoningEnd(block_id="r1"))

    h.handle_event(TextStart(block_id="t1"))
    h.handle_event(TextDelta(block_id="t1", delta="I'll check"))
    h.handle_event(TextEnd(block_id="t1"))

    h.handle_event(ToolStart(tool_call_id="tc1", tool_name="search"))
    h.handle_event(ToolArgsDelta(tool_call_id="tc1", delta='{"q":"test"}'))
    m = h.handle_event(ToolEnd(tool_call_id="tc1"))

    assert len(m.parts) == 3
    assert isinstance(m.parts[0], ReasoningPart)
    assert isinstance(m.parts[1], TextPart)
    assert isinstance(m.parts[2], ToolPart)
    assert all(
        p.state == "done"
        for p in m.parts
        if isinstance(p, (TextPart, ToolPart, ReasoningPart))
    )


def test_multiple_tool_calls() -> None:
    """Parallel tool calls in one message."""
    h = StreamHandler(message_id="m1")
    h.handle_event(ToolStart(tool_call_id="tc1", tool_name="read_file"))
    h.handle_event(ToolStart(tool_call_id="tc2", tool_name="list_files"))

    m = h.handle_event(ToolArgsDelta(tool_call_id="tc1", delta='{"path":"a.py"}'))
    # Both tools should be in parts
    tool_parts = [p for p in m.parts if isinstance(p, ToolPart)]
    assert len(tool_parts) == 2
    # tc1 has args, tc2 is empty
    assert tool_parts[0].tool_args == '{"path":"a.py"}'
    assert tool_parts[1].tool_args == ""

    h.handle_event(ToolArgsDelta(tool_call_id="tc2", delta='{"dir":"."}'))
    h.handle_event(ToolEnd(tool_call_id="tc1"))
    m = h.handle_event(ToolEnd(tool_call_id="tc2"))
    assert all(
        p.state == "done"
        for p in m.parts
        if isinstance(p, (TextPart, ToolPart, ReasoningPart))
    )


# -- MessageDone -----------------------------------------------------------


def test_message_done_finalizes_all() -> None:
    h = StreamHandler(message_id="m1")
    h.handle_event(TextStart(block_id="t1"))
    h.handle_event(TextDelta(block_id="t1", delta="hello"))
    # Don't send TextEnd -- MessageDone should finalize everything
    m = h.handle_event(MessageDone(finish_reason="end_turn"))
    part = m.parts[0]
    assert isinstance(part, TextPart)
    assert part.state == "done"
    assert m.is_done


# -- Message properties propagate ------------------------------------------


def test_message_id_propagates() -> None:
    h = StreamHandler(message_id="custom-id")
    m = h.handle_event(TextStart(block_id="b1"))
    assert m.id == "custom-id"


def test_deltas_only_on_active_blocks() -> None:
    """Delta should be None on inactive blocks, present only on active."""
    h = StreamHandler(message_id="m1")
    h.handle_event(TextStart(block_id="t1"))
    h.handle_event(TextDelta(block_id="t1", delta="first"))
    h.handle_event(TextEnd(block_id="t1"))

    h.handle_event(TextStart(block_id="t2"))
    m = h.handle_event(TextDelta(block_id="t2", delta="second"))

    text_parts = [p for p in m.parts if isinstance(p, TextPart)]
    assert text_parts[0].delta is None  # t1 is done
    assert text_parts[1].delta == "second"  # t2 is active
