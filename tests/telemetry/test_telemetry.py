"""Telemetry: event sequence, run_id, enable/disable, error capture."""

from __future__ import annotations

import dataclasses
from collections.abc import Generator
from typing import Any

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.telemetry.events import (
    RunFinishEvent,
    RunStartEvent,
    TelemetryEvent,
    ToolCallFinishEvent,
    ToolCallStartEvent,
)

from ..conftest import MockLLM, text_msg, tool_msg

# ── Recording handler ────────────────────────────────────────────


class RecordingHandler:
    """Captures all telemetry events in order for assertions."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def handle(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def of_type(self, cls: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, cls)]


@pytest.fixture
def handler() -> Generator[RecordingHandler]:
    h = RecordingHandler()
    ai.telemetry.enable(h)
    yield h
    ai.telemetry.disable()


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


# ── Event sequence: text-only run ────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_run_events(handler: RecordingHandler) -> None:
    """Simplest run emits RunStart, StepStart, StepFinish, RunFinish."""

    async def root(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(llm, messages=ai.make_messages(user="Hi"), tools=[])

    result = ai.run(root, MockLLM([[text_msg("Hello!")]]))
    [m async for m in result]

    types = [type(e).__name__ for e in handler.events]
    assert types == [
        "RunStartEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "RunFinishEvent",
    ]
    assert handler.of_type(RunFinishEvent)[0].error is None


# ── Event sequence: tool call run ────────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_events(handler: RecordingHandler) -> None:
    """Tool-calling run emits tool events between steps with correct payloads."""

    async def root(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm, messages=ai.make_messages(user="Double 5"), tools=[double]
        )

    llm = MockLLM(
        [
            [tool_msg(tc_id="tc-1", name="double", args='{"x": 5}')],
            [text_msg("10")],
        ]
    )
    result = ai.run(root, llm)
    [m async for m in result]

    types = [type(e).__name__ for e in handler.events]
    assert types == [
        "RunStartEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "ToolCallStartEvent",
        "ToolCallFinishEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "RunFinishEvent",
    ]

    tc_start = handler.of_type(ToolCallStartEvent)[0]
    tc_finish = handler.of_type(ToolCallFinishEvent)[0]
    assert tc_start.tool_name == "double"
    assert tc_start.args == '{"x": 5}'
    assert tc_finish.result == 10
    assert tc_finish.error is None
    assert tc_finish.duration_ms >= 0


# ── run_id contextvar ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_id_available_during_run() -> None:
    """get_run_id() returns a non-empty ID inside a handler during a run."""
    captured: str = ""

    class Capture:
        def handle(self, event: TelemetryEvent) -> None:
            nonlocal captured
            if isinstance(event, RunStartEvent):
                captured = ai.telemetry.get_run_id()

    ai.telemetry.enable(Capture())
    try:

        async def root(llm: ai.LanguageModel) -> ai.StreamResult:
            return await ai.stream_loop(
                llm, messages=ai.make_messages(user="Hi"), tools=[]
            )

        result = ai.run(root, MockLLM([[text_msg("Hello!")]]))
        [m async for m in result]
        assert len(captured) == 16
    finally:
        ai.telemetry.disable()


# ── enable / disable lifecycle ───────────────────────────────────


@pytest.mark.asyncio
async def test_disable_reverts_to_noop() -> None:
    """disable() stops events from reaching the handler."""
    handler = RecordingHandler()
    ai.telemetry.enable(handler)

    async def root(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(llm, messages=ai.make_messages(user="Hi"), tools=[])

    result = ai.run(root, MockLLM([[text_msg("Hello!")]]))
    [m async for m in result]
    assert len(handler.of_type(RunStartEvent)) == 1

    ai.telemetry.disable()
    handler.events.clear()

    result = ai.run(root, MockLLM([[text_msg("Hello!")]]))
    [m async for m in result]
    assert len(handler.events) == 0


# ── User-emitted custom events ──────────────────────────────────


@pytest.mark.asyncio
async def test_user_emitted_custom_event(handler: RecordingHandler) -> None:
    """ai.telemetry.handle() delivers user events to the active handler."""

    @dataclasses.dataclass(frozen=True, slots=True)
    class CustomEvent(TelemetryEvent):
        message: str

    async def root(llm: ai.LanguageModel) -> ai.StreamResult:
        ai.telemetry.handle(CustomEvent(message="hello"))
        return await ai.stream_loop(llm, messages=ai.make_messages(user="Hi"), tools=[])

    result = ai.run(root, MockLLM([[text_msg("Hello!")]]))
    [m async for m in result]

    custom = [e for e in handler.events if isinstance(e, CustomEvent)]
    assert len(custom) == 1
    assert custom[0].message == "hello"


# ── Error capture ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_error_in_finish_event(handler: RecordingHandler) -> None:
    """RunFinishEvent captures the error when the root function raises."""

    async def root(llm: ai.LanguageModel) -> None:
        raise ValueError("boom")

    result = ai.run(root, MockLLM([]))
    with pytest.raises(ExceptionGroup):
        [m async for m in result]

    run_finish = handler.of_type(RunFinishEvent)[0]
    assert run_finish.error is not None
