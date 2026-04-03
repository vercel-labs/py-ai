"""@stream decorator: context requirement, replay, queue submission."""

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.agents2.streams import StreamResult
from vercel_ai_sdk.types import messages

from ..conftest import MOCK_MODEL, mock_llm, text_msg


class _Weather(pydantic.BaseModel):
    city: str
    temperature: float


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
    last = r.last_message
    assert last is not None
    assert last.id == "m2"
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
    mock_llm([[text_msg("hi")]])
    with pytest.raises(ValueError, match="No Runtime context"):
        await ai.stream_step(
            MOCK_MODEL,
            ai.make_messages(user="test"),
        )


# -- @stream replays from checkpoint --------------------------------------


@pytest.mark.asyncio
async def test_stream_step_replays_from_checkpoint() -> None:
    """stream_step inside Agent.run with a checkpoint replays without calling LLM."""

    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> ai.StreamResult:
        return await ai.stream_step(agent.model, msgs)

    # First run
    mock_llm([[text_msg("Hi")]])
    r1 = my_agent.run(ai.make_messages(user="hello"))
    [msg async for msg in r1]
    cp = r1.checkpoint

    # Replay
    llm2 = mock_llm([])
    r2 = my_agent.run(ai.make_messages(user="hello"), checkpoint=cp)
    [msg async for msg in r2]
    assert llm2.call_count == 0


# -- StreamResult.output ---------------------------------------------------


def test_stream_result_output_from_last_message() -> None:
    """StreamResult.output delegates to the last message's StructuredOutputPart."""
    m = messages.Message(
        id="m1",
        role="assistant",
        parts=[
            messages.TextPart(text="{}", state="done"),
            messages.StructuredOutputPart(
                data={"city": "SF", "temperature": 62.0},
                output_type_name=f"{_Weather.__module__}.{_Weather.__qualname__}",
            ),
        ],
    )
    r = StreamResult(messages=[text_msg("streaming..."), m])
    assert r.output is not None
    assert r.output.city == "SF"
