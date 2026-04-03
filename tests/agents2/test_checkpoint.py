"""Checkpoint replay, hook cancellation/resolution, serialization."""

import asyncio
from typing import Any, ClassVar

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.agents2.checkpoint import Checkpoint, HookEvent, StepEvent, ToolEvent

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_msg


@ai.hook
class Approval(pydantic.BaseModel):
    cancels_future: ClassVar[bool] = True
    granted: bool


# -- Replay ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_replay_skips_llm() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> ai.StreamResult:
        return await ai.stream_step(agent.model, msgs)

    llm1 = mock_llm([[text_msg("Hi there!")]])
    result1 = my_agent.run(ai.make_messages(system="test", user="hello"))
    [msg async for msg in result1]
    assert llm1.call_count == 1

    cp = result1.checkpoint
    llm2 = mock_llm([])
    result2 = my_agent.run(ai.make_messages(system="test", user="hello"), checkpoint=cp)
    [msg async for msg in result2]
    assert llm2.call_count == 0


@pytest.mark.asyncio
async def test_tool_replay_skips_execution() -> None:
    execution_count = 0

    @ai.tool
    async def counting_tool(x: int) -> int:
        """Counts calls."""
        nonlocal execution_count
        execution_count += 1
        return x + 1

    my_agent = ai.agent(model=MOCK_MODEL, tools=[counting_tool])

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> ai.StreamResult:
        result = await ai.stream_step(agent.model, msgs, agent.tools)
        if result.tool_calls:
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )
        return result

    mock_llm([[tool_msg(tc_id="tc-1", name="counting_tool", args='{"x": 5}')]])
    result1 = my_agent.run(ai.make_messages(system="t", user="go"))
    [msg async for msg in result1]
    assert execution_count == 1
    assert result1.checkpoint.tools[0].result == 6

    execution_count = 0
    mock_llm([])
    result2 = my_agent.run(
        ai.make_messages(system="t", user="go"), checkpoint=result1.checkpoint
    )
    [msg async for msg in result2]
    assert execution_count == 0


# -- Hooks -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_cancellation_pending() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> Any:
        await ai.stream_step(agent.model, msgs)
        return await Approval.create("my_approval", metadata={"tool": "test"})  # type: ignore[attr-defined]

    mock_llm([[text_msg("OK")]])
    result = my_agent.run(ai.make_messages(system="t", user="go"))
    msgs = [msg async for msg in result]
    assert "my_approval" in result.pending_hooks
    hook_msgs = [m for m in msgs if any(isinstance(p, ai.HookPart) for p in m.parts)]
    assert hook_msgs[0].parts[0].status == "pending"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_hook_resolution_on_reentry() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> Any:
        await ai.stream_step(agent.model, msgs)
        return await Approval.create("my_approval")  # type: ignore[attr-defined]

    resp = [text_msg("OK")]
    mock_llm([resp])
    result1 = my_agent.run(ai.make_messages(system="t", user="go"))
    [msg async for msg in result1]
    cp = result1.checkpoint

    Approval.resolve("my_approval", {"granted": True})  # type: ignore[attr-defined]
    mock_llm([])
    result2 = my_agent.run(ai.make_messages(system="t", user="go"), checkpoint=cp)
    [msg async for msg in result2]
    assert len(result2.pending_hooks) == 0
    assert result2.checkpoint.hooks[-1].label == "my_approval"


@pytest.mark.asyncio
async def test_parallel_hooks_all_collected() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        await ai.stream_step(agent.model, msgs)

        async def a() -> Any:
            return await Approval.create("hook_a")  # type: ignore[attr-defined]

        async def b() -> Any:
            return await Approval.create("hook_b")  # type: ignore[attr-defined]

        async with asyncio.TaskGroup() as tg:
            tg.create_task(a())
            tg.create_task(b())

    mock_llm([[text_msg("OK")]])
    result = my_agent.run(ai.make_messages(system="t", user="go"))
    [msg async for msg in result]
    assert {"hook_a", "hook_b"} <= set(result.pending_hooks)


@pytest.mark.asyncio
async def test_parallel_hooks_resolve_on_reentry() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> Any:
        await ai.stream_step(agent.model, msgs)

        async def a() -> Any:
            return await Approval.create("hook_a")  # type: ignore[attr-defined]

        async def b() -> Any:
            return await Approval.create("hook_b")  # type: ignore[attr-defined]

        async with asyncio.TaskGroup() as tg:
            ta = tg.create_task(a())
            tb = tg.create_task(b())
        return ta.result(), tb.result()

    resp = [text_msg("OK")]
    mock_llm([resp])
    result1 = my_agent.run(ai.make_messages(system="t", user="go"))
    [msg async for msg in result1]
    cp = result1.checkpoint

    Approval.resolve("hook_a", {"granted": True})  # type: ignore[attr-defined]
    Approval.resolve("hook_b", {"granted": False})  # type: ignore[attr-defined]
    mock_llm([])
    result2 = my_agent.run(ai.make_messages(system="t", user="go"), checkpoint=cp)
    [msg async for msg in result2]
    assert len(result2.pending_hooks) == 0


# -- Serialization ---------------------------------------------------------


def test_checkpoint_serialization_roundtrip() -> None:
    cp = Checkpoint(
        steps=[
            StepEvent(
                index=0,
                messages=[
                    ai.Message(
                        id="m1",
                        role="assistant",
                        parts=[ai.TextPart(text="hi")],
                    )
                ],
            )
        ],
        tools=[ToolEvent(tool_call_id="tc-1", result=42)],
        hooks=[HookEvent(label="h1", resolution={"granted": True})],
    )
    cp2 = Checkpoint.model_validate(cp.model_dump())
    assert cp2.steps[0].index == 0
    assert cp2.tools[0].result == 42
    assert cp2.hooks[0].label == "h1"
