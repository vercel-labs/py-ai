"""Checkpoint replay, hook cancellation/resolution, serialization."""

import asyncio

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core.checkpoint import Checkpoint, HookEvent, StepEvent, ToolEvent

from ..conftest import MockLLM, text_msg, tool_msg


@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool


# -- Replay ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_replay_skips_llm():
    async def graph(llm: ai.LanguageModel):
        return await ai.stream_step(
            llm, messages=ai.make_messages(system="test", user="hello")
        )

    llm1 = MockLLM([[text_msg("Hi there!")]])
    result1 = ai.run(graph, llm1)
    [msg async for msg in result1]
    assert llm1.call_count == 1

    cp = result1.checkpoint
    llm2 = MockLLM([])
    result2 = ai.run(graph, llm2, checkpoint=cp)
    [msg async for msg in result2]
    assert llm2.call_count == 0


@pytest.mark.asyncio
async def test_tool_replay_skips_execution():
    execution_count = 0

    @ai.tool
    async def counting_tool(x: int) -> int:
        """Counts calls."""
        nonlocal execution_count
        execution_count += 1
        return x + 1

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        if result.tool_calls:
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )
        return result

    llm1 = MockLLM([[tool_msg(tc_id="tc-1", name="counting_tool", args='{"x": 5}')]])
    result1 = ai.run(graph, llm1)
    [msg async for msg in result1]
    assert execution_count == 1
    assert result1.checkpoint.tools[0].result == 6

    execution_count = 0
    result2 = ai.run(graph, MockLLM([]), checkpoint=result1.checkpoint)
    [msg async for msg in result2]
    assert execution_count == 0


# -- Hooks -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_cancellation_pending():
    async def graph(llm: ai.LanguageModel):
        await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        return await Approval.create("my_approval", metadata={"tool": "test"})

    result = ai.run(graph, MockLLM([[text_msg("OK")]]), cancel_on_hooks=True)
    msgs = [msg async for msg in result]
    assert "my_approval" in result.pending_hooks
    hook_msgs = [m for m in msgs if any(isinstance(p, ai.HookPart) for p in m.parts)]
    assert hook_msgs[0].parts[0].status == "pending"


@pytest.mark.asyncio
async def test_hook_resolution_on_reentry():
    async def graph(llm: ai.LanguageModel):
        await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        return await Approval.create("my_approval")

    resp = [text_msg("OK")]
    result1 = ai.run(graph, MockLLM([resp]), cancel_on_hooks=True)
    [msg async for msg in result1]
    cp = result1.checkpoint

    Approval.resolve("my_approval", {"granted": True})
    result2 = ai.run(graph, MockLLM([]), checkpoint=cp)
    [msg async for msg in result2]
    assert len(result2.pending_hooks) == 0
    assert result2.checkpoint.hooks[-1].label == "my_approval"


@pytest.mark.asyncio
async def test_parallel_hooks_all_collected():
    async def graph(llm: ai.LanguageModel):
        await ai.stream_step(llm, ai.make_messages(system="t", user="go"))

        async def a():
            return await Approval.create("hook_a")

        async def b():
            return await Approval.create("hook_b")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(a())
            tg.create_task(b())

    result = ai.run(graph, MockLLM([[text_msg("OK")]]), cancel_on_hooks=True)
    [msg async for msg in result]
    assert {"hook_a", "hook_b"} <= set(result.pending_hooks)


@pytest.mark.asyncio
async def test_parallel_hooks_resolve_on_reentry():
    async def graph(llm: ai.LanguageModel):
        await ai.stream_step(llm, ai.make_messages(system="t", user="go"))

        async def a():
            return await Approval.create("hook_a")

        async def b():
            return await Approval.create("hook_b")

        async with asyncio.TaskGroup() as tg:
            ta = tg.create_task(a())
            tb = tg.create_task(b())
        return ta.result(), tb.result()

    resp = [text_msg("OK")]
    result1 = ai.run(graph, MockLLM([resp]), cancel_on_hooks=True)
    [msg async for msg in result1]
    cp = result1.checkpoint

    Approval.resolve("hook_a", {"granted": True})
    Approval.resolve("hook_b", {"granted": False})
    result2 = ai.run(graph, MockLLM([]), checkpoint=cp)
    [msg async for msg in result2]
    assert len(result2.pending_hooks) == 0


# -- Serialization ---------------------------------------------------------


def test_checkpoint_serialization_roundtrip():
    cp = Checkpoint(
        steps=[
            StepEvent(
                index=0,
                messages=[
                    {
                        "id": "m1",
                        "role": "assistant",
                        "parts": [{"type": "text", "text": "hi"}],
                    }
                ],
            )
        ],
        tools=[ToolEvent(tool_call_id="tc-1", result=42)],
        hooks=[HookEvent(label="h1", resolution={"granted": True})],
    )
    cp2 = Checkpoint.deserialize(cp.serialize())
    assert cp2.steps[0].index == 0
    assert cp2.tools[0].result == 42
    assert cp2.hooks[0].label == "h1"
