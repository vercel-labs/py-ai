"""StreamHandler: event accumulation, state transitions, message building."""

from __future__ import annotations

from collections.abc import Sequence

from ai.models.core.helpers import streaming
from ai.types import events, messages


def _only[T](items: Sequence[object], typ: type[T]) -> T:
    matches = [item for item in items if isinstance(item, typ)]
    assert len(matches) == 1
    return matches[0]


def test_text_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")

    out = h.handle_event(streaming.TextStart(block_id="b1"))
    assert isinstance(out[0], events.TextStart)
    assert out[0].block_id == "b1"

    out = h.handle_event(streaming.TextDelta(block_id="b1", delta="Hello"))
    delta = _only(out, events.TextDelta)
    assert delta.chunk == "Hello"
    assert delta.block_id == "b1"

    out = h.handle_event(streaming.TextDelta(block_id="b1", delta=" world"))
    delta = _only(out, events.TextDelta)
    assert delta.chunk == " world"

    out = h.handle_event(streaming.TextEnd(block_id="b1"))
    assert isinstance(out[0], events.TextEnd)
    assert out[0].block_id == "b1"
    assert not any(isinstance(event, events.TextDelta) for event in out)

    out = h.handle_event(streaming.MessageDone(finish_reason="end_turn"))
    msg = _only(out, events.MessageEnd).message
    assert msg.text == "Hello world"


def test_reasoning_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ReasoningStart(block_id="r1"))

    out = h.handle_event(streaming.ReasoningDelta(block_id="r1", delta="thinking"))
    delta = _only(out, events.ReasoningDelta)
    assert delta.chunk == "thinking"

    out = h.handle_event(streaming.ReasoningEnd(block_id="r1", signature="sig123"))
    end = _only(out, events.ReasoningEnd)
    assert end.signature == "sig123"

    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message
    assert msg.reasoning == "thinking"
    part = msg.parts[0]
    assert isinstance(part, messages.ReasoningPart)
    assert part.signature == "sig123"


def test_tool_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")

    out = h.handle_event(
        streaming.ToolStart(tool_call_id="tc1", tool_name="get_weather")
    )
    start = _only(out, events.ToolStart)
    assert start.tool_name == "get_weather"

    out = h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"ci'))
    delta = _only(out, events.ToolDelta)
    assert delta.chunk == '{"ci'

    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='ty":"London"}'))

    out = h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))
    assert isinstance(out[0], events.ToolEnd)
    assert not any(isinstance(event, events.ToolDelta) for event in out)

    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message
    tc = msg.tool_calls[0]
    assert tc.tool_name == "get_weather"
    assert tc.tool_args == '{"city":"London"}'


def test_reasoning_then_text_then_tool() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ReasoningStart(block_id="r1"))
    h.handle_event(streaming.ReasoningDelta(block_id="r1", delta="Let me think"))
    h.handle_event(streaming.ReasoningEnd(block_id="r1"))

    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="I'll check"))
    h.handle_event(streaming.TextEnd(block_id="t1"))

    h.handle_event(streaming.ToolStart(tool_call_id="tc1", tool_name="search"))
    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"q":"test"}'))
    out = h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))
    assert isinstance(out[0], events.ToolEnd)

    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message
    assert len(msg.parts) == 3
    assert isinstance(msg.parts[0], messages.ReasoningPart)
    assert isinstance(msg.parts[1], messages.TextPart)
    assert isinstance(msg.parts[2], messages.ToolCallPart)


def test_multiple_tool_calls() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ToolStart(tool_call_id="tc1", tool_name="read_file"))
    h.handle_event(streaming.ToolStart(tool_call_id="tc2", tool_name="list_files"))

    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"path":"a.py"}'))
    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc2", delta='{"dir":"."}'))
    h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))
    out = h.handle_event(streaming.ToolEnd(tool_call_id="tc2"))
    assert isinstance(out[0], events.ToolEnd)
    assert out[0].tool_call_id == "tc2"

    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message
    tool_parts = [p for p in msg.parts if isinstance(p, messages.ToolCallPart)]
    assert [p.tool_args for p in tool_parts] == ['{"path":"a.py"}', '{"dir":"."}']


def test_message_done_finalizes_all() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="hello"))

    out = h.handle_event(streaming.MessageDone(finish_reason="end_turn"))
    final = _only(out, events.MessageEnd)
    assert final.message.text == "hello"


def test_message_done_propagates_usage() -> None:
    usage = messages.Usage(input_tokens=10, output_tokens=20)
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="hi"))

    h.handle_event(streaming.TextEnd(block_id="t1"))
    final = _only(h.handle_event(streaming.MessageDone(usage=usage)), events.MessageEnd)
    assert final.usage is not None
    assert final.usage.input_tokens == 10
    assert final.message.usage is not None
    assert final.message.usage.total_tokens == 30


def test_deltas_only_on_active_blocks() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="first"))
    h.handle_event(streaming.TextEnd(block_id="t1"))

    h.handle_event(streaming.TextStart(block_id="t2"))
    out = h.handle_event(streaming.TextDelta(block_id="t2", delta="second"))

    deltas = [event for event in out if isinstance(event, events.TextDelta)]
    assert len(deltas) == 1
    assert deltas[0].block_id == "t2"
    assert deltas[0].chunk == "second"


def test_file_event_accumulates() -> None:
    h = streaming.StreamHandler(message_id="m1")
    out = h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="iVBORw0KGgo=")
    )
    assert out == []

    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message
    assert len(msg.images) == 1
    assert msg.images[0].media_type == "image/png"
    assert msg.images[0].data == "iVBORw0KGgo="


def test_file_event_with_text() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="Here is your image:"))
    h.handle_event(streaming.TextEnd(block_id="t1"))
    h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="iVBORw0KGgo=")
    )
    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message

    assert msg.text == "Here is your image:"
    assert len(msg.images) == 1


def test_multiple_file_events() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="png_data")
    )
    h.handle_event(
        streaming.FileEvent(block_id="f2", media_type="image/jpeg", data="jpeg_data")
    )
    msg = _only(h.handle_event(streaming.MessageDone()), events.MessageEnd).message

    assert [p.media_type for p in msg.images] == ["image/png", "image/jpeg"]
