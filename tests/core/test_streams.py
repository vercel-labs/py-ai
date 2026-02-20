"""@stream decorator: context requirement, replay, queue submission."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core import messages
from vercel_ai_sdk.core.streams import StreamResult

from ..conftest import MockLLM, text_msg


# -- StreamResult properties -----------------------------------------------


def test_stream_result_empty() -> None:
    r = StreamResult()
    assert r.last_message is None
    assert r.tool_calls == []
    assert r.text == ""


def test_stream_result_last_message() -> None:
    m1 = text_msg("first", id="m1")
    m2 = text_msg("second", id="m2")
    r = StreamResult(messages=[m1, m2])
    assert r.last_message.id == "m2"
    assert r.text == "second"


def test_stream_result_tool_calls() -> None:
    m = messages.Message(
        id="m1",
        role="assistant",
        parts=[
            messages.ToolPart(
                tool_call_id="tc1", tool_name="t", tool_args="{}", state="done"
            ),
            messages.ToolPart(
                tool_call_id="tc2", tool_name="u", tool_args="{}", state="done"
            ),
        ],
    )
    r = StreamResult(messages=[m])
    assert len(r.tool_calls) == 2


# -- @stream requires Runtime context -------------------------------------


@pytest.mark.asyncio
async def test_stream_outside_run_raises() -> None:
    """@stream-decorated fn called without ai.run() should raise."""
    with pytest.raises(ValueError, match="No Runtime context"):
        await ai.stream_step(
            MockLLM([[text_msg("hi")]]),
            ai.make_messages(user="test"),
        )


# -- @stream replays from checkpoint --------------------------------------


@pytest.mark.asyncio
async def test_stream_step_replays_from_checkpoint() -> None:
    """stream_step inside ai.run with a checkpoint replays without calling LLM."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_step(llm, ai.make_messages(user="hello"))

    # First run
    llm1 = MockLLM([[text_msg("Hi")]])
    r1 = ai.run(graph, llm1)
    [msg async for msg in r1]
    cp = r1.checkpoint

    # Replay
    llm2 = MockLLM([])
    r2 = ai.run(graph, llm2, checkpoint=cp)
    [msg async for msg in r2]
    assert llm2.call_count == 0
