"""Runtime: stream_loop end-to-end, execute_tool, multi-turn, Runtime injection."""

import asyncio

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core import messages
from vercel_ai_sdk.core.runtime import Runtime

from ..conftest import MockLLM, text_msg, tool_msg


# -- Tool definitions for tests --------------------------------------------


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


@ai.tool
async def concat(a: str, b: str) -> str:
    """Concatenate strings."""
    return a + b


# -- stream_loop: single turn (no tools) ----------------------------------


@pytest.mark.asyncio
async def test_stream_loop_text_only() -> None:
    """stream_loop with no tool calls returns after one LLM call."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="Hi"),
            tools=[double],
        )

    llm = MockLLM([[text_msg("Hello!")]])
    result = ai.run(graph, llm)
    msgs = [m async for m in result]
    assert llm.call_count == 1
    assert any(m.text == "Hello!" for m in msgs)


# -- stream_loop: tool call + follow-up -----------------------------------


@pytest.mark.asyncio
async def test_stream_loop_tool_then_text() -> None:
    """stream_loop calls tool, feeds result back, gets final text."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="Double 5"),
            tools=[double],
        )

    call1 = [tool_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    call2 = [text_msg("The answer is 10.")]
    llm = MockLLM([call1, call2])

    result = ai.run(graph, llm)
    msgs = [m async for m in result]
    assert llm.call_count == 2
    # Tool should have been executed: 5 * 2 = 10
    tool_results = [
        m for m in msgs if m.tool_calls and m.tool_calls[0].status == "result"
    ]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_calls[0].result == 10


# -- stream_loop: multiple tool calls in one message ----------------------


@pytest.mark.asyncio
async def test_stream_loop_parallel_tools() -> None:
    """LLM returns two tool calls in one message; both execute."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="Double 3 and 7"),
            tools=[double],
        )

    two_tools = messages.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages.ToolPart(
                tool_call_id="tc-1",
                tool_name="double",
                tool_args='{"x": 3}',
                status="pending",
                state="done",
            ),
            messages.ToolPart(
                tool_call_id="tc-2",
                tool_name="double",
                tool_args='{"x": 7}',
                status="pending",
                state="done",
            ),
        ],
    )
    call2 = [text_msg("6 and 14", id="msg-2")]
    llm = MockLLM([[two_tools], call2])

    result = ai.run(graph, llm)
    msgs = [m async for m in result]
    assert llm.call_count == 2
    # Both tools should have results
    tool_result_msgs = [
        m
        for m in msgs
        if m.tool_calls and any(tc.status == "result" for tc in m.tool_calls)
    ]
    assert len(tool_result_msgs) >= 1


# -- stream_loop: multi-turn (tool -> tool -> text) -----------------------


@pytest.mark.asyncio
async def test_stream_loop_multi_turn() -> None:
    """LLM calls a tool, then calls another tool, then returns text."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="Concat then double"),
            tools=[double, concat],
        )

    turn1 = [
        tool_msg(tc_id="tc-1", name="concat", args='{"a": "hello", "b": " world"}')
    ]
    turn2 = [tool_msg(tc_id="tc-2", name="double", args='{"x": 3}', id="msg-2")]
    turn3 = [text_msg("Done: hello world, 6", id="msg-3")]
    llm = MockLLM([turn1, turn2, turn3])

    result = ai.run(graph, llm)
    [m async for m in result]
    assert llm.call_count == 3


# -- execute_tool: missing tool raises ------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_missing_raises() -> None:
    """execute_tool with unknown tool name raises ValueError (wrapped in ExceptionGroup by TaskGroup)."""
    tc = messages.ToolPart(
        tool_call_id="tc-1", tool_name="nonexistent_tool_zzz", tool_args="{}"
    )

    async def graph(llm: ai.LanguageModel) -> None:
        await ai.execute_tool(tc)

    result = ai.run(graph, MockLLM([]))
    with pytest.raises(ExceptionGroup) as exc_info:
        [m async for m in result]
    assert any(isinstance(e, ValueError) for e in exc_info.value.exceptions)


# -- execute_tool: Runtime injection ---------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_injects_runtime() -> None:
    """Tools with a Runtime parameter get the active runtime injected."""
    received_rt = None

    @ai.tool
    async def introspect(query: str, rt: Runtime) -> str:
        """Tool that inspects runtime."""
        nonlocal received_rt
        received_rt = rt
        return "ok"

    async def graph(llm: ai.LanguageModel) -> None:
        result = await ai.stream_step(llm, ai.make_messages(user="go"))
        if result.tool_calls:
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )

    call = [tool_msg(tc_id="tc-1", name="introspect", args='{"query": "test"}')]
    result = ai.run(graph, MockLLM([call]))
    [m async for m in result]
    assert received_rt is not None
    assert isinstance(received_rt, Runtime)


# -- execute_tool: result updates ToolPart in message ----------------------


@pytest.mark.asyncio
async def test_execute_tool_updates_message() -> None:
    """After execute_tool, the ToolPart in the message has status=result."""

    async def graph(llm: ai.LanguageModel) -> None:
        result = await ai.stream_step(llm, ai.make_messages(user="go"))
        if result.tool_calls:
            msg = result.last_message
            for tc in result.tool_calls:
                await ai.execute_tool(tc, message=msg)
            # Verify the tool part was mutated
            assert msg.tool_calls[0].status == "result"
            assert msg.tool_calls[0].result == 10

    call = [tool_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    result = ai.run(graph, MockLLM([call]))
    [m async for m in result]


# -- Checkpoint records tools from stream_loop -----------------------------


@pytest.mark.asyncio
async def test_stream_loop_checkpoint_records_tools() -> None:
    """stream_loop's tool executions are recorded in the checkpoint."""

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="Double 4"),
            tools=[double],
        )

    call1 = [tool_msg(tc_id="tc-1", name="double", args='{"x": 4}')]
    call2 = [text_msg("8", id="msg-2")]
    llm = MockLLM([call1, call2])

    result = ai.run(graph, llm)
    [m async for m in result]

    cp = result.checkpoint
    assert any(t.tool_call_id == "tc-1" and t.result == 8 for t in cp.tools)
